document.getElementById('mushroom-form').addEventListener('submit', function (e) {
    e.preventDefault();

    let formData = new FormData(this);
    let formObject = {};
    formData.forEach((value, key) => formObject[key] = value);

    // Basic validation: Check if all fields are filled
    for (let key in formObject) {
        if (formObject[key].trim() === '') {
            alert(`Please fill in the ${key.replace('-', ' ')} field.`);
            return;
        }
    }

    fetch('https://http://ecs170mushroomclassification.s3-website.us-east-2.amazonaws.com/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formObject)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('result').innerText = `Result: ${data.message}`;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerText = 'An error occurred. Please try again.';
        });
});
