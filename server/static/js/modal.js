$(window).ready(function(){
$(".boton").wrapInner('<div class=botontext></div>');
    
    $(".botontext").clone().appendTo( $(".boton") );
    
    $(".boton").append('<span class="twist"></span><span class="twist"></span><span class="twist"></span><span class="twist"></span>');
    
    $(".twist").css("width", "25%").css("width", "+=3px");
});

$(document).ready(function() {
    var panels = $('.user-infos');
    var panelsButton = $('.dropdown-user');
    panels.hide();

    //Click dropdown
    panelsButton.click(function() {
        //get data-for attribute
        var dataFor = $(this).attr('data-for');
        var idFor = $(dataFor);

        //current button
        var currentButton = $(this);
        idFor.slideToggle(400, function() {
            //Completed slidetoggle
            if(idFor.is(':visible'))
            {
                currentButton.html('<i class="glyphicon glyphicon-chevron-up text-muted"></i>');
            }
            else
            {
                currentButton.html('<i class="glyphicon glyphicon-chevron-down text-muted"></i>');
            }
        })
    });
    $('[data-toggle="tooltip"]').tooltip();

});
$('#modal').on('show.bs.modal', function () {
       $(this).find('.modal-body').css({
              width:'auto', 
              height:'auto', 
              'max-height':'100%'
       });
});



 // $(function() {
 //       // bind 'myForm' and provide a simple callback function
 //       $('#myForm').ajaxForm(function() {
 //           alert("Thank you for choosing filter!");
           
 //       });
 //     });

function getval(sel)
{
    var start=sel.value;
    console.log(start);
}
    