function showPostForm(){
            var e = document.getElementById('contactForm');
            if(e.style.display == null || e.style.display == "none") {
                e.style.display = "block";
            } else {
                e.style.display = "none";
            }    
        }

function showReservations(){
            var e = document.getElementById('contactForm2');
            if(e.style.display == null || e.style.display == "none") {
                e.style.display = "block";
            } else {
                e.style.display = "none";
            }    
        }


function showInvitations(){
            var e = document.getElementById('contactForm3');
            if(e.style.display == null || e.style.display == "none") {
                e.style.display = "block";
            } else {
                e.style.display = "none";
            }    
        }


function validateForm() {
            
            var bas_rating= document.forms["change"]["basketballrating"].value;
            var sw_rating= document.forms["change"]["swimmingrating"].value;
            var sq_rating= document.forms["change"]["squashrating"].value;
            var st_rating= document.forms["change"]["strengthrating"].value;
            var car_rating= document.forms["change"]["cardiorating"].value;
            if( bas_rating== "") {
            alert("You did not select basketballrating score!");
            return false;
            }; 
            if( sw_rating== "") {
            alert("You did not select swimmingrating score!");
            return false;
            }; 
            if( sq_rating== "") {
            alert("You did not select squashrating score!");
            return false;
            }; 
            if( st_rating== "") {
            alert("You did not select strengthrating score!");
            return false;
            }; 
            if( car_rating== "") {
            alert("You did not select cardiorating score!");
            return false;
            }; 

          }
