async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    const response = await fetch('/api/chat', {  // Use /api/ endpoint which will be rewritten to Render backend
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ input_text: userInput })
    });

    const data = await response.json();
    document.getElementById("chatbox").innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
    document.getElementById("chatbox").innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
    document.getElementById("userInput").value = '';
}
