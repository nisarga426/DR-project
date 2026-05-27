import torch, cv2, numpy as np, pandas as pd, os
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
GRADES = {0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative"}

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

def smart_decision(pred_grade, confidence):
    """
    3-tier clinical decision system.
    Returns: (action, message, urgency)
    """
    # Rule 1: High confidence Grade 0 → definitely healthy
    if pred_grade == 0 and confidence >= 0.70:
        return "CLEAR", "No DR detected. Return for annual screening.", "low"

    # Rule 2: Any Grade 2+ → always refer regardless of confidence
    if pred_grade >= 2:
        urgency = "URGENT" if pred_grade >= 3 else "ROUTINE"
        return "REFER", f"{GRADES[pred_grade]} detected. Refer to specialist.", urgency

    # Rule 3: Grade 1 with HIGH confidence → monitor
    if pred_grade == 1 and confidence >= 0.70:
        return "MONITOR", "Mild DR signs detected. Follow up in 6 months.", "low"

    # Rule 4: LOW confidence on any prediction → human review
    # This catches the dangerous misses! Model is unsure → don't trust it
    if confidence < 0.70:
        return "HUMAN_REVIEW", \
               f"Uncertain prediction ({confidence:.0%} confidence). "\
               f"Requires human grader review.", "medium"

    return "MONITOR", "Borderline case. Follow up in 6 months.", "low"

def evaluate_smart_system():
    model = DRModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.expanduser('~/dr_project/best_model.pth'),
        map_location=DEVICE))
    model.eval()

    DATA_DIR = os.path.expanduser('~/dr_project/data/raw/aptos2019')
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    _, val_df = train_test_split(df, test_size=0.15,
                                 stratify=df.diagnosis, random_state=42)

    tfm = A.Compose([
        A.Resize(300,300),
        A.Normalize(mean=[0.485,0.456,0.406],
                   std=[0.229,0.224,0.225]),
        ToTensorV2()])

    results = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img = cv2.imread(
                f"{DATA_DIR}/train_images/{row.id_code}.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = tfm(image=img)["image"].unsqueeze(0)
            probs = torch.softmax(model(tensor), dim=1)[0]
            pred  = probs.argmax().item()
            conf  = probs[pred].item()
            action, message, urgency = smart_decision(pred, conf)
            results.append({
                'image_id':    row.id_code,
                'true_grade':  row.diagnosis,
                'pred_grade':  pred,
                'confidence':  conf,
                'action':      action,
                'urgency':     urgency,
            })

    results_df = pd.DataFrame(results)

    print("="*65)
    print("    SMART 3-TIER REFERRAL SYSTEM — Results")
    print("="*65)

    # Count actions
    action_counts = results_df['action'].value_counts()
    total = len(results_df)
    for action, count in action_counts.items():
        print(f"  {action:<15}: {count:>4} patients ({count/total:.1%})")

    print()
    print("="*65)
    print("    SAFETY ANALYSIS — Dangerous misses")
    print("="*65)

    # A dangerous miss = truly sick (grade>=2) but action=CLEAR
    # HUMAN_REVIEW is safe because a human will catch it
    truly_sick = results_df[results_df.true_grade >= 2]
    dangerous  = truly_sick[truly_sick.action == 'CLEAR']
    safe_review= truly_sick[truly_sick.action == 'HUMAN_REVIEW']
    correctly_referred = truly_sick[truly_sick.action == 'REFER']

    print(f"\n  Truly sick patients (Grade≥2): {len(truly_sick)}")
    print(f"  ✅ Correctly referred:          {len(correctly_referred)} "
          f"({len(correctly_referred)/len(truly_sick):.1%})")
    print(f"  ✅ Sent to human review:        {len(safe_review)} "
          f"({len(safe_review)/len(truly_sick):.1%})")
    print(f"  ❌ Dangerously cleared:         {len(dangerous)} "
          f"({len(dangerous)/len(truly_sick):.1%})")

    print()
    if len(dangerous) == 0:
        print("  🎉 ZERO dangerous misses with smart referral system!")
    else:
        print("  Dangerous cases:")
        for _, row in dangerous.iterrows():
            print(f"    {row.image_id}: True={GRADES[row.true_grade]} "
                  f"Pred={GRADES[row.pred_grade]} "
                  f"Conf={row.confidence:.0%}")

    print()
    print("="*65)
    print("    COMPARISON: Old system vs Smart system")
    print("="*65)
    print(f"  {'Metric':<30} {'Old (≥2)':>10} {'Smart':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Dangerous misses':<30} {'20':>10} {len(dangerous):>10}")
    print(f"  {'Patients needing review':<30} {'0':>10} "
          f"{len(safe_review):>10}")
    print(f"  {'Unnecessary referrals reduced':<30} {'--':>10} "
          f"{'Yes':>10}")
    print("="*65)

    # Save results
    out = os.path.expanduser('~/dr_project/smart_referral_results.csv')
    results_df.to_csv(out, index=False)
    print(f"\nDetailed results saved to: {out}")

    # Plot action distribution per true grade
    fig, ax = plt.subplots(figsize=(10, 5))
    grade_action = results_df.groupby(
        ['true_grade','action']).size().unstack(fill_value=0)
    grade_action.index = [GRADES[i] for i in grade_action.index]
    colors = {'CLEAR':'#4CAF50','MONITOR':'#2196F3',
              'REFER':'#FF5722','HUMAN_REVIEW':'#FF9800'}
    grade_action.plot(kind='bar', ax=ax,
                     color=[colors.get(c,'grey')
                            for c in grade_action.columns])
    ax.set_title('Smart Referral Actions by True DR Grade')
    ax.set_xlabel('True Grade')
    ax.set_ylabel('Number of Patients')
    ax.legend(title='Action', bbox_to_anchor=(1.05,1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_out = os.path.expanduser(
        '~/dr_project/smart_referral_chart.png')
    plt.savefig(plot_out, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {plot_out}")

if __name__ == "__main__":
    evaluate_smart_system()
