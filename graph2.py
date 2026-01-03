import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
knn_values = [0.8341, 0.8000, 0.8932, 0.8440]
colors = ['#22223b', '#4a4e69', '#9a8c98', '#c9ada7']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, knn_values, color=colors, edgecolor='black')

plt.xlabel("Evaluation Metrics", fontweight='bold')
plt.ylabel("Score", fontweight='bold')
plt.title("KNN Classification Performance Metrics (k=5)", fontweight='bold', fontsize=14)

plt.ylim(0, 1)
plt.grid(True, axis='y', alpha=0.3)

for bar, score in zip(bars, knn_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,f'{score:.3f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.show()
