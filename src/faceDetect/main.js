const drawingUtils = window; 
const mpFaceDetection = window;

const FACE_BOX_SCALAR = 0.4;

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');


function getAbsoluteBoundingBox(normalizedBox, absW, absH) {
	let nW = normalizedBox.width;
	let nH = normalizedBox.height;
	let nX = normalizedBox.xCenter;
	let nY = normalizedBox.yCenter;
	let w = nW * absW;
	let h = nH * absH;
	let x = nX * absW - (w/2);
	let y = nY * absH - (h/2);
	return [x, y, w, h];
}

function expandBoundingBox(absoluteBoundingBox, scalar) {
	let [x, y, w, h] = absoluteBoundingBox;
	let wExpansion = w * scalar;
	let hExpansion = h * scalar;
	let newX = x - wExpansion;
	let newY = y - hExpansion;
	let newW = w + wExpansion * 2;
	let newH = h + hExpansion * 2;
	return [newX, newY, newW, newH];
}



function onResults(results) {
  // Hide the spinner.
  document.body.classList.add('loaded');

  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  
  if(results.detections[0]) {
 
	  
	  let boundingBox = getAbsoluteBoundingBox(results.detections[0].boundingBox, canvasElement.width, canvasElement.height);
	      boundingBox = expandBoundingBox(boundingBox, FACE_BOX_SCALAR);
	      
	  canvasCtx.beginPath();
	  canvasCtx.rect(...boundingBox);
	  canvasCtx.stroke();	 	 	 
	   
	  let faceImageData = canvasCtx.getImageData(...boundingBox);
	  canvasCtx.putImageData(faceImageData, 0, 0);
	  
   
	  
  }
  
  //let faceImg = canvasCtx.getImageData(...results.detections[0].boundingBox);
    
  /*
  if (results.detections.length > 0) {
    drawingUtils.drawRectangle(
        canvasCtx, results.detections[0].boundingBox,
        {color: 'blue', lineWidth: 4, fillColor: '#00000000'});
    drawingUtils.drawLandmarks(canvasCtx, results.detections[0].landmarks, {
      color: 'red',
      radius: 5,
    });
  }
  */
  canvasCtx.restore();
}

const faceDetection = new mpFaceDetection.FaceDetection({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4/${file}`;
}});

faceDetection.setOptions({
  selfieMode: true,
  model: "short",
  minDetectionConfidence: 0.65
});

faceDetection.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceDetection.send({image: videoElement});
  },
  width: 1280,
  height: 720
});

camera.start();


