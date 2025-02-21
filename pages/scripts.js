let model;

// Load the trained model
async function loadONNXModel() {
    const session = await ort.InferenceSession.create("model/knn_model.onnx");

    // Example input (Make sure values match your feature order)
    let inputTensor = new ort.Tensor("float32", new Float32Array([
        bmi, glucose, sex, age, bloodPressure, skinThickness, insulin, parseFloat(diabetesPedigree)
    ]), [1, 8]);

    // Run model prediction
    let results = await session.run({ float_input: inputTensor });
    let output = results.output.data[0]; // 0 or 1 (Diabetes Prediction)

    document.getElementById("result").innerText = output > 0.5 ? "Diabetes Detected" : "No Diabetes";
}


loadModel();  // Load the model when the page loads

async function predictDiabetes() {
    let skinThickness = parseFloat(document.getElementById("skinThickness").value) || 0;
    let insulin = parseFloat(document.getElementById("insulin").value) || 0;
    let diabetesPedigree = document.getElementById("diabetesPedigree").value || '0.54';
    let bmi = parseFloat(document.getElementById("bmi").value);
    let glucose = parseFloat(document.getElementById("glucose").value);
    let sex = document.getElementById("sex").value === "Male" ? 1 : 0;
    let age = parseFloat(document.getElementById("age").value);
    let bloodPressure = parseFloat(document.getElementById("bloodPressure").value);

    if (isNaN(bmi) || isNaN(glucose) || isNaN(age) || isNaN(bloodPressure)) {
        alert("Please enter valid values.");
        return;
    }

    document.getElementById("result").innerText = "Processing...";

    // Prepare input data
    let inputTensor = tf.tensor2d([[bmi, glucose, sex, age, bloodPressure, skinThickness, insulin, parseFloat(diabetesPedigree)]]);
    
    // Make prediction
    let prediction = model.predict(inputTensor);
    let result = await prediction.data();

    document.getElementById("result").innerText = result[0] > 0.5 ? "Diabetes Detected" : "No Diabetes";
}
