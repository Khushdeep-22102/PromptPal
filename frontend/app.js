async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    
    // Make sure the user input is not empty before sending the request
    if (!userInput.trim()) {
        return;
    }

    const response = await fetch('/api/chat', {  // Endpoint will be rewritten to the backend URL
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ input_text: userInput })
    });

    const data = await response.json();

    // Display user input and bot response in the chatbox
    const chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
    chatbox.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
    
    // Clear the input field
    document.getElementById("userInput").value = '';

    // Auto-scroll to the bottom of the chatbox to show the latest messages
    chatbox.scrollTop = chatbox.scrollHeight;
}
