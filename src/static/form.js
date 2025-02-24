$('#imageUpload').on('change', function () {
    let file = this.files[0];
    let reader = new FileReader();

    reader.onload = function (event) {
        let base64String = event.target.result;
        $('#preview').attr('src', base64String);
        $('#preview').css('display', 'block');
        //console.log(base64String);
        let img = new Image();
        img.src = base64String;
        let aspectRatio = img.width / img.height;
        $('#preview').attr('height', Math.round(parseFloat($('#preview').attr('height')) / aspectRatio));
    };

    reader.readAsDataURL(file);

    $('.beforeSubmit').each((i, e) => {
        e.hidden = false;
    });

    $('#imageCaption').val('');
    $('#imageQuestion').val('');
});

$("#submitButton").on("click", function() {
    var img = $('#preview').attr('src');
    var question = $('#imageQuestion').val();
    $.ajax({
        url: '/infer',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({'img': img, 'question': (question === '') ? null : question}),
        success: function(response) {
            $('#imageCaption').val(response.result);
            $('#imageCaption').show();
        },
        error: function(error) {
            console.log(error);
        }
    });
});