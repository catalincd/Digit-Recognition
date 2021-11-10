var model = null;
var canvas = document.getElementById("drawBoard");
var testCanvas = document.getElementById("testBoard");
var canvasScale = 1.0;
var ctx = canvas.getContext("2d");
var pos = {
    x: 0,
    y: 0
};

const min = (a, b) => (a < b ? a : b);

var canvasSize = 280;

function invert(imageData) {
    for (var i = 0; i < imageData.data; i++) {
        if ((i + 1) % 4 != 0)
            data[i] = 255 - data[i];
    }
    return imageData;
}

function downscale(imageData) {
    var chunkSize = canvasSize / 28;
    var chunkNum = canvasSize / chunkSize;

    var img = ctx.createImageData(28, 28);

    var chunkCount = 0;

    for (i = 0; i < canvasSize; i += chunkSize) {
        for (j = 0; j < canvasSize; j += chunkSize) {
            var sum = 0;

            for (var x = 0; x < chunkSize; x++) {
                for (var y = 0; y < chunkSize; y++) {
                    sum += imageData.data[(i + x) + (j + y) * canvasSize];
                }
            }


            sum /= chunkSize * chunkSize;
            sum = 255 - sum;

            img.data[chunkCount * 4 + 0] = sum;
            img.data[chunkCount * 4 + 1] = sum;
            img.data[chunkCount * 4 + 2] = sum;
            img.data[chunkCount * 4 + 3] = 255;
            chunkCount++;
        }
    }

    console.log(chunkCount)
    console.log(img)

    return img;
}


async function init() {
    resize();
    clearCanvas();
    console.log("LOADING MODEL");
    model = await tf.loadLayersModel('http://localhost:3000/model/model.json');
    console.log(model);
    console.log("LOADED MODEL");
}

async function predict() {

    $(".digit").css({
        "color": "white"
    })

    const pred = await tf.tidy(() => {


        let img = tf.browser.fromPixels(getImageData(), 1);
        img = img.reshape([1, 28, 28]);
        img = tf.cast(img, 'float32');

        const output = model.predict(img);
        var thisPred = Array.from(output.dataSync());
        displayPredictions(thisPred);
    });
}

function clearCanvas() {
    $(".digit").css({
        "color": "#333"
    })
    displayPredictions([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fill();
}

function resize() {
    canvasScale = 280.0 / ($("#drawBoard").width())
}

function setPosition(e) {
    pos.x = (e.clientX - canvas.offsetLeft) * canvasScale;
    pos.y = (e.clientY - canvas.offsetTop) * canvasScale;
}


function draw(e) {
    if (e.buttons !== 1) return;

    ctx.beginPath();

    ctx.filter = "blur(3px)";
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#FFF';

    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);

    ctx.stroke();
}



function getImageData() {
    var smallCanvas = document.createElement('canvas');
    var smallContext = smallCanvas.getContext("2d");
    smallContext.scale(0.1, 0.1);
    //smallContext.filter = "blur(1px)";
    smallContext.drawImage(canvas, 0, 0);

    if (testCanvas !== null)
        testCanvas.getContext("2d").drawImage(smallCanvas, 0, 0);

    return smallContext.getImageData(0, 0, 28, 28);
}


function displayPredictions(arr) {
    for (var i = 0; i < arr.length; i++) {
        $(`.p${i}`).css({
            "maxHeight": `${arr[i] * 100}%`
        });
    }
}

var predictionWrapper = $("#predictionWrapper");

for (var i = 0; i < 10; i++) {
    predictionWrapper.append(`  <div class="prediction" id="${i}">
                                    <div class="predictionFiller  p${i}"></div>
                                    <p class="digit">${i}</p>
                                </div>`);
}





window.addEventListener("load", init);
document.addEventListener('mousemove', draw);
document.addEventListener('mousedown', setPosition);
document.addEventListener('mouseenter', setPosition);