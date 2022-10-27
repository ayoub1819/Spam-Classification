// Example POST method implementation:
let email = document.getElementById('email-body');
let loaders = document.getElementsByClassName('spinner-border');
let badges = document.getElementsByClassName('badge');
let models = ['svm','knn','dt','rf']




async function postData(url = '', data = {}) {

    // Default options are marked with *
    const response = await fetch(url, {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data) 
    });
    return response.json(); 
  }
  
  

function classify(){
    showLoaders();
        
    postData('http://127.0.0.1:5000/test', {
    "email": email.value,
    })
    .then((data) => {
      hideLoaders();
      displayResults(data);
});
}



function showLoaders(){
  for(let i = 0; i <loaders.length;i++){
    loaders[i].classList.remove('d-none');
    badges[i].classList.add('d-none');
  }
}

function hideLoaders(){
  for(let i = 0; i <badges.length;i++){
    loaders[i].classList.add('d-none');
   
  }
}

function displayResults(data){
  let result = '';
  for(let i = 0; i <badges.length;i++){
    result = data[models[i]];
    badges[i].textContent = data[models[i]];
    if(data[models[i]] === "spam"){
      badges[i].classList.remove('bg-success');
      badges[i].classList.add('bg-danger');
    }else{
      badges[i].classList.add('bg-success');
      badges[i].classList.remove('bg-danger');
    }

    badges[i].classList.remove('d-none');
    
  }
}