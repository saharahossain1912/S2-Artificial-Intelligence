import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
logistic_values = [0.7902, 0.7459, 0.8835, 0.8089]
colors=['#6f1d1b','#bb9457', '#432818','#99582a']

plt.figure(figsize=(10, 6))
bars=plt.bar(metrics, logistic_values, color=colors, edgecolor='black')

plt.xlabel("Evaluation Metrics", fontweight='bold')
plt.ylabel("Score", fontweight='bold')
plt.title("Logistic Regression Performance Metrics", fontweight='bold')

plt.ylim(0, 1)
plt.grid(True, axis='y', alpha=0.3)
for bar, score in zip(bars, logistic_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,f'{score:.3f}', ha='center',
             va='bottom', fontweight='bold', fontsize=11)
    
plt.tight_layout()
plt.show()