import React, {useRef, useState} from "react"
import "./App.css"
import {load, ModelConfig} from "@tensorflow-models/posenet"
import "@tensorflow/tfjs-backend-webgl"
import Webcam from "react-webcam";

let loadPosenet = async () => {
    const modelConfig: ModelConfig = {
        architecture: "ResNet50",
        outputStride: 32,
        inputResolution: {width: 800, height: 600},
        quantBytes: 2,
        multiplier: 1
    }
    let model = await load(modelConfig)
    console.log("Posenet model loaded...")
    return model
}

function App() {
    const [model, setModel] = useState(loadPosenet())
    const webcamRef = useRef(<Webcam/>)
    return (
        <div className="App">
            <header className="App-header">
                <Webcam ref={webcamRef}
                width={800}
                height={600}
                zindex={9}/>
            </header>
        </div>
    );
}

export default App;