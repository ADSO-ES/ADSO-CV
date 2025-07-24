from src.cvPipeline import run_detection

# Paths
image_path = "data/test.png"
yolo_model_path = "models/yolo_sign_obstacles.pt"
cnn_model_path = "models/CNN_sign_obstacles.h5"
output_csv_path = "outputs/combined_yolo_cnn_results.csv"

# Run detection
df = run_detection(image_path, yolo_model_path, cnn_model_path)

# Output
print(df.head())
df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")

