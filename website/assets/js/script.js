var srcBase64;
var termynal = new Termynal('#termynal');
function allowDrop(ev) {
	ev.preventDefault();
}

function drag(ev) {
	ev.dataTransfer.setData('text', ev.target.id);
}

function drop(ev) {
	ev.preventDefault();
	var id = ev.dataTransfer.getData('text');
	console.log(id);
	uploadDemo(id);
}

function uploadDemo(id) {
	srcBase64 = getBase64Image(document.getElementById(id));
	document.getElementById('result-page').classList.replace('d-flex', 'd-none');
	document.getElementById('xray-placeholder').style.backgroundImage = "url('" + srcBase64 + "')";
	document.getElementById('chooseFile').innerText = id;
	document.getElementById('textoverimg').innerText = 'Upload It';
	sendFile();
}

function getBase64Image(img) {
	var canvas = document.createElement('canvas');
	canvas.width = img.naturalWidth;
	canvas.height = img.naturalHeight;
	var ctx = canvas.getContext('2d');
	ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
	return canvas.toDataURL('image/png');
}

$(document).ready(function() {
	$('#numberOfCases').text(Erkrankungen);
});

function isValid(file) {
	return isValidExtension(file) && isValidSize(file);
}

function isValidExtension(file) {
	var filename = file.name;
	var filesize = file.size; // in Byte
	var fileExtension = filename
		.split('.')
		.pop()
		.toLowerCase();

	// There are more extensions/formats supportet, see:
	// https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html
	var allowedExtensions = ['bmp', 'gif', 'jpg', 'jpeg', 'jpe', 'jfif', 'png', 'tiff', 'tga'];
	if ($.inArray(fileExtension, allowedExtensions) == -1) {
		$('#fileExtensionAlert').show();
		return false;
	} else {
		$('#fileExtensionAlert').hide();
		return true;
	}
}

function isValidSize(file) {
	var filesize = file.size; // in Byte

	// Maximum file size is set to 10MB, can be changed if necessarily
	if (filesize > 10 * 1000000) {
		$('#fileSizeAlert').show();
		return false;
	} else {
		$('#fileSizeAlert').hide();
		return true;
	}
}

function readyToUpload() {
	var selectedFiles = document.getElementById('inputFileToLoad').files;
	document.getElementById('result-page').classList.replace('d-flex', 'd-none');
	if (selectedFiles.length > 0) {
		var file = selectedFiles[0];
		console.log(file);
		if (isValid(file)) {
			document.querySelector('#uploadButton').disabled = false;
			document.querySelector('#uploadButton').classList.remove('btn-secondary');
			document.querySelector('#uploadButton').classList.add('btn-primary');

			var fileReader = new FileReader();
			fileReader.onload = function(fileLoadedEvent) {
				srcBase64 = fileLoadedEvent.target.result;
				document.getElementById('xray-placeholder').style.backgroundImage = "url('" + srcBase64 + "')";
				document.getElementById('chooseFile').innerText = file.name;
				document.getElementById('textoverimg').innerText = '';
			};
		} else {
			document.querySelector('#uploadButton').disabled = true;
			document.querySelector('#uploadButton').classList.remove('btn-primary');
			document.querySelector('#uploadButton').classList.add('btn-secondary');
		}
	}
	fileReader.readAsDataURL(file);
}
var resultText='';
function sendFile() {
	document.getElementById('textoverimg').innerText = 'Uploading ...';
	document.querySelector('.input-group').style.display = 'none';
	document.getElementById('progressContainer').style.display = 'flex';
	url = 'https://europe-west3-rezaamini.cloudfunctions.net/mode06';
	//url='https://lit-shelf-32128.herokuapp.com/predict';

	var base64Data = srcBase64.substr(srcBase64.indexOf(',') + 1); // cutt the data:image/png;base64, prefix away
	var sendData = {
		type: 'xray',
		image: base64Data
	};

	$.ajax({
		type: 'POST',
		url: 'https://europe-west3-rezaamini.cloudfunctions.net/mode06',
		//url:'https://lit-shelf-32128.herokuapp.com/predict',

		data: JSON.stringify(sendData),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		crossDomain: true,
		success: function(res) {
			//alert(res);
			console.log(res);
			var possibility = 100 * Math.sqrt(parseFloat(res));
			resX(possibility);


			document.getElementById('covidSlider').style.left = Math.round(200 * res - 1);
			document.getElementById('textoverimg').innerText = "";
			document.querySelector('.input-group').style.display = 'flex';
			document.getElementById('progressContainer').style.display = 'none';
			document.getElementById('uploadProgress').style.width = 0;
			document.getElementById('result-page').classList.replace('d-none', 'd-flex');
		},
		xhr: function() {
			var appXhr = $.ajaxSettings.xhr();
			if (appXhr.upload) {
				appXhr.upload.addEventListener(
					'progress',
					function(e) {
						if (e.lengthComputable) {
							var currentProgress = (e.loaded / e.total) * 100; // Amount uploaded in percent
							document.getElementById('uploadProgress').style.width = Math.round(currentProgress) + '%';
							if (currentProgress == 100) {
								document.getElementById('textoverimg').innerText = 'Diagnosing ...';
							}
						}
					},
					false
				);
			}
			return appXhr;
		}
	});
}



function resX(result) {
			var covidStatus = result > 50 ? 'Positive' : 'Negative';
			resultText1 = ' Possibility : \t '+ Math.round(result) +'%\n' + ' Patient State  : \t '   + covidStatus;
			document.getElementById('textimg1').innerText = resultText1;



if (result>=0 && result<=30) {
  			resultText2 = "Status: \t"+" Healthy";
			document.getElementById('textimg2').innerText = resultText2;

			resultText3 = " No need to Hospitalization, stay in home";
			document.getElementById('textimg3').innerText = resultText3;
}


else if (result>30 && result<=50) {
  			resultText2 = "Status: \t"+"Relatively healthy";
			document.getElementById('textimg2').innerText = resultText2;

				resultText3 = "Quarantine in home ";
			document.getElementById('textimg3').innerText = resultText3;
} else if (result>=51 && result<=65){
				resultText2 =  "Status: \t"+ " Need to Hospitalization "
			document.getElementById('textimg2').innerText = resultText2;

			resultText3 = "We suggest to stay in hospital ";
			document.getElementById('textimg3').innerText = resultText3;
} else if (result>66 && result<80){
				resultText2 =  "Status: \t"+ "Serious illness"
			document.getElementById('textimg2').innerText = resultText2;

			resultText3 = "Need to go under more care";
			document.getElementById('textimg3').innerText = resultText3;

}
else{
				resultText2 =  "Status: \t"+" Dangerous "
			document.getElementById('textimg2').innerText = resultText2;

			resultText3 = "we suggest to use ICU ";
			document.getElementById('textimg3').innerText = resultText3;
}


}


function printContent(el){
		var restorepage=document.body.innerHTML;
		var printcontent=document.getElementById(el).innerHTML;
		document.body.innerHTML=printcontent;
		window.print();
		document.body.innerHTML=restorepage;


}
