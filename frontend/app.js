async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    const response = await fetch('http://localhost:8000/chat', {  // Added /chat here
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ input_text: userInput })  // Ensure this matches the backend parameter
    });

    const data = await response.json();
    document.getElementById("chatbox").innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
    document.getElementById("chatbox").innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
    document.getElementById("userInput").value = '';
}
