let model;

// Load TensorFlow.js model
async function loadModel() {
    try {
        console.log("🔍 Attempting to load model from: pages/model/model.json");
        model = await tf.loadLayersModel('pages/model/model.json');
        console.log("✅ Model loaded successfully!");

        // Log input shape to check if it's properly defined
        console.log("📏 Model Input Shape:", model.inputs[0].shape);

    } catch (error) {
        console.error("❌ Error loading the model:", error);
    }
}

// Load model when page loads
window.onload = async () => {
    await loadModel();
};


async function predictDiabetes() {
    if (!model) {
        alert("Model is still loading, please wait...");
        return;
    }

    // Get user input values
    let age = parseFloat(document.getElementById("age").value);
    let pregnancies = parseFloat(document.getElementById("pregnancies").value);
    let bmi = parseFloat(document.getElementById("bmi").value);
    let glucose = parseFloat(document.getElementById("glucose").value);
    let bloodPressure = parseFloat(document.getElementById("bloodPressure").value);
    let hbA1c = parseFloat(document.getElementById("hbA1c").value);
    let ldl = parseFloat(document.getElementById("ldl").value);
    let hdl = parseFloat(document.getElementById("hdl").value);
    let triglycerides = parseFloat(document.getElementById("triglycerides").value);
    let waistCircumference = parseFloat(document.getElementById("waistCircumference").value);
    let hipCircumference = parseFloat(document.getElementById("hipCircumference").value);
    let whr = parseFloat(document.getElementById("whr").value);
    let familyHistory = document.getElementById("familyHistory").checked ? 1 : 0; // Binary
    let dietType = parseInt(document.getElementById("dietType").value); // 0, 1, or 2
    let hypertension = document.getElementById("hypertension").checked ? 1 : 0; // Binary
    let medicationUse = document.getElementById("medicationUse").checked ? 1 : 0; // Binary

    // Validate required fields
    if (isNaN(age) || isNaN(pregnancies) || isNaN(bmi) || isNaN(glucose) || isNaN(bloodPressure) || 
        isNaN(hbA1c) || isNaN(ldl) || isNaN(hdl) || isNaN(triglycerides) || isNaN(waistCircumference) || 
        isNaN(hipCircumference) || isNaN(whr) || isNaN(dietType)) {
        alert("Please enter valid numerical values for all required fields.");
        return;
    }

    document.getElementById("result").innerText = "Processing...";

    // Prepare the input tensor with correct shape [1, 16]
    let inputTensor = tf.tensor2d([[
        age, pregnancies, bmi, glucose, bloodPressure, hbA1c, ldl, hdl, triglycerides,
        waistCircumference, hipCircumference, whr, familyHistory, dietType, hypertension, medicationUse
    ]], [1, 16]);

    // Make a prediction
    let prediction = model.predict(inputTensor);
    let result = await prediction.data(); // Extract prediction result as an array

    console.log("🔍 Model Prediction Output:", result);

    // Since model output has 2 units, we take the second probability for diabetes detection
    let diabetesProbability = result[1]; // Model predicts probability of class 1 (diabetes)

    // Display result
    document.getElementById("result").innerText = diabetesProbability > 0.5 
        ? `Diabetes Detected (Risk Score: ${(diabetesProbability * 100).toFixed(2)}%)` 
        : `No Diabetes (Risk Score: ${(diabetesProbability * 100).toFixed(2)}%)`;
}

