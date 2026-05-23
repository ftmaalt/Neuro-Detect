# NeuroDetect - MRI Brain Tumor Detection and Classification using Image Processing & Deep Learning 

## 📌 About the Project:
NeuroDetect is a University Undergraduate Graduation Project developed to help reduce the time between brain tumor detection and treatment, as early diagnosis can significantly improve and accelarate the treament and recovery process. The project also addresses some of the drawbacks of manual diagnosis by providing AI-assisted MRI scan classification.
The application analyzes brain MRI images and classifies them using multiclass classification into one of the following categories:
- Glioma Tumor 
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

This Web application was developed for educational and research purposes only and is not intended to replace professional medical diagnosis or clinical evaluation.

 ## ❓ How it Works
 1. Upload a brain MRI scan through the Diagnostic Portal  
2. The image is resized and preprocessed  
3. The trained Deep Learning model analyzes the scan  
4. The system generates:
   - Predicted classification
   - Confidence score
   - Probability breakdown
   
## 🎛️ Model Information

| Feature | Details |
|---|---|
| Model Type | Convolutional Neural Network (CNN) |
| Transfer Learning | EfficientNetB3 |
| Input Size | 256 × 256 |
| Classification Type | Multiclass |
| Classes | 4 |
| Final Accuracy | 98.8% |

## 📃 Datasets Used
The model was trained using publicly available Kaggle MRI datasets:
- [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


## ⚙️ Tech Tools and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Model training and implementation |
| Streamlit | Web application framework |
| NumPy | Numerical processing |
| Pillow (PIL) | Image processing |
| Jupyter Notebook | Model development environment |

## 🚀 How to Run

### Steps
1. Open [NeuroDetect](https://neurodetectai.com/) on any web browser
2. Agree to the usage terms
3. Enter the Diagnostic Portal
4. Upload a valid brain MRI scan  
5. Click **Initiate Neural Diagnostic**  
6. Review the generated results

## ‼️Limitations
- Model is trained on publicly available datasets only
- Not clinically validated
- Limited dataset diversity
- Requires MRI image quality consistency

## 👥 Project Team

**Developed By**
- Yusra Husain Haji
- Fatima Mohamed Hasan 

---ITSE498---
