async function sendMessage() {
    // Get user input from the input field
    const userInput = document.getElementById("userInput").value;

    // Ensure the user input is not empty or just spaces before making the request
    if (!userInput.trim()) {
        return;
    }

    try {
        // Make a POST request to the backend API endpoint with the user's input
        const response = await fetch('/api/chat', {  // Replace '/api/chat' with the backend URL if necessary
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ input_text: userInput })
        });

        // Parse the JSON response from the backend
        const data = await response.json();

        // Log the response for debugging purposes
        console.log("Response from backend:", data);

        // Check if the response contains a valid 'response' field
        if (data && data.response) {
            // Get the chatbox element and append the user input and bot response
            const chatbox = document.getElementById("chatbox");
            chatbox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
            chatbox.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
            
            // Clear the input field after sending the message
            document.getElementById("userInput").value = '';

            // Auto-scroll to the bottom of the chatbox to show the latest messages
            chatbox.scrollTop = chatbox.scrollHeight;
        } else {
            // Log an error if the response format is incorrect
            console.error("Response format error: 'response' field is missing.");
        }

    } catch (error) {
        // Catch and log any errors that occur during the fetch operation
        console.error("Error during fetch request:", error);
    }
}
