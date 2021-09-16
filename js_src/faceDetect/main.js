const drawingUtils = window; 
const mpFaceDetection = window;

const FACE_BOX_SCALAR = 0.4;

const videoElement = document.getElementsByClassName('input_video')[0];
const inCanvas = document.getElementsByClassName('input_canvas')[0];
const inCtx = inCanvas.getContext('2d');

const outCanvas = document.getElementsByClassName('output_canvas')[0];
const outCtx = inCanvas.getContext('2d');

var FACE;

var OLD_MASK;
var MASK;

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

function extract_face(results) {

  inCtx.clearRect(0, 0, inCanvas.width, inCanvas.height);
  inCtx.drawImage(results.image, 0, 0, inCanvas.width, inCanvas.height);
  
  if(results.detections[0]) {
	let boundingBox = getAbsoluteBoundingBox(results.detections[0].boundingBox, inCanvas.width, inCanvas.height);
		boundingBox = expandBoundingBox(boundingBox, FACE_BOX_SCALAR);
	      
	inCtx.beginPath();
	inCtx.rect(...boundingBox);
	inCtx.stroke();
	inCtx.closePath();	 	 	 
	   
	let faceImageData = inCtx.getImageData(...boundingBox);	  
	FACE = faceImageData;
  }  
}

function set_mask(results) {
	MASK = results.segmentationMask;
	//outCtx.clearRect(0, 0, outCanvas.width, outCanvas.height);
	//outCtx.drawImage(results.segmentationMask, 0, 0, outCanvas.width, outCanvas.height);
	//outCtx.globalCompositeOperation = 'source-in';
    //outCtx.fillStyle = '#00FF00';
    //outCtx.fillRect(0, 0, outCanvas.width, outCanvas.height);
}

function render_loop() {
	let mask;
	
	if(MASK) {
		mask = MASK;
		OLD_MASK = MASK;
	} else if (OLD_MASK) {
		mask = OLD_MASK;
	}
	
	if(MASK) {
		console.log("loop!");
		outCtx.clearRect(0, 0, outCanvas.width, outCanvas.height);
		outCtx.drawImage(mask, 0, 0, outCanvas.width, outCanvas.height);
		outCtx.globalCompositeOperation = 'source-in';
		outCtx.fillStyle = '#00FF00';
		outCtx.fillRect(0, 0, outCanvas.width, outCanvas.height);
	}
	
    window.requestAnimationFrame(render_loop);		
}

const faceDetection = new mpFaceDetection.FaceDetection({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4/${file}`;
}});

faceDetection.setOptions({
  selfieMode: true,
  model: "short",
  minDetectionConfidence: 0.65
});
faceDetection.onResults(extract_face);

const selfieSegmentation = new SelfieSegmentation({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
}});

selfieSegmentation.setOptions({
  modelSelection: 1,
});
selfieSegmentation.onResults(set_mask);


const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceDetection.send({image: videoElement});
    if(FACE) { await selfieSegmentation.send({image: FACE}) };
  },
  width: 640,				
  height: 360
});

camera.start();
setTimeout(function() {	render_loop(); }, 5000);


