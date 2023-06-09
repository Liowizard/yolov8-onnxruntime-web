import Webcam from "react-webcam";
import React, { useState, useRef, useEffect } from "react";
import cv, { log } from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const webRef = useRef(null);
  const [base64Img, setBase64Img] = useState("");
  const [imgD, setimgD] = useState("");
  
  useEffect(() => {
    const interval = setInterval(() => {
      let base64data;
      if (webRef.current === null) {
        return;
      } else {

        base64data = webRef.current.getScreenshot();
      }
  

      const binaryData = window.atob(base64data.split(",")[1]);
      const buffer = new ArrayBuffer(binaryData.length);
      const view = new Uint8Array(buffer);
      for (let i = 0; i < binaryData.length; i++) {
        view[i] = binaryData.charCodeAt(i);
      }
  
      // Create a Blob from the ArrayBuffer
      const blob = new Blob([buffer], { type: "image/jpeg" });
  
      // Convert Blob to base64 format
      const reader = new FileReader();
      reader.onloadend = () => {
        setBase64Img(reader.result);
      };
      reader.readAsDataURL(blob);
      const img=blob
      setimgD(img)
      handleAaa(img)
    }, 50);
  
    return () => clearInterval(interval); // Clear the interval on component unmount
  }, []);
  
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({
    text: "Loading OpenCV.js",
    progress: null,
  });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Configs
  const modelName = "yolov8n.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading YOLOv8 Segmentation model", setLoading] // logger
    );
    const yolov8 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov8.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    );
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );

    await yolov8.run({ images: tensor });
    setSession({ net: yolov8, nms: nms });
    setLoading(null);
  };

  const handleAaa =(imgD)=>{
    if (image) {
                  URL.revokeObjectURL(image);
                  setImage(null);
                }
                const url = URL.createObjectURL(imgD); // create image url
                imageRef.current.src = url; // set image source
                setImage(url);
  }

  return (
    <div className="App">
      <div className="APP">
        <Webcam ref={webRef} />
        {/* {base64Img&&<img src={`${base64Img}`} alt="Webcam" />} */}
      </div>
      {loading && (
        <Loader>
          {loading.progress
            ? `${loading.text} - ${loading.progress}%`
            : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>YOLOv8 Object Detection App</h1>
        <p>
          YOLOv8 object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }
          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
