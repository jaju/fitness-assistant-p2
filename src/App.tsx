import React from "react"
import logo from "./logo.svg"
import "./App.css"
import * as tf from "@tensorflow/tfjs"
import {Logs, Sequential, Tensor} from "@tensorflow/tfjs";

let doTraining = async (model: Sequential, xs: tf.Tensor, ys: tf.Tensor, epochs: number = 200) => {
    let logLogger = async (epoch: number, logs: Logs | undefined) => {
        if (logs && epoch % 10 === 0) {
            console.log("Epoch: " + epoch + ", loss: " + logs["loss"])
        }
    }
    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
            onEpochEnd: logLogger
        }
    })
    console.log(history.params)
}

let initModel = () => {
    const model: Sequential = tf.sequential()
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))
    model.compile({optimizer: tf.train.sgd(0.01), loss: "meanSquaredError"})
    return model
}

const trainingData = [
    [-10.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 19.0],
    [-21.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 37.0]
]

const model = initModel()
let handleRunTraining = async () => {
    let [trainXs, trainYs] = trainingData
    let xs = tf.tensor2d(trainXs, [trainXs.length, 1])
    let ys = tf.tensor2d(trainYs, [trainYs.length, 1])
    await doTraining(model, xs, ys)
        .then(() => {
            let input = 10
            model.summary()
            const prediction = (model.predict(tf.tensor2d([input], [1, 1])) as Tensor).dataSync()
            console.log("Predicted value for '" + input + "' : " + prediction[0])
            tf.dispose(prediction)
        })
}

function App() {
    return (
        <div className="App">
            <header className="App-header">
                <img src={logo} className="App-logo" alt="logo"/>
                <p>
                    Edit <code>src/App.tsx</code> and save to reload.
                </p>
                <a
                    className="App-link"
                    href="https://reactjs.org"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Learn React
                </a>
                <br/>
                <button onClick={handleRunTraining}>Run Training</button>
                (and watch the console output)
            </header>
        </div>
    );
}

export default App;