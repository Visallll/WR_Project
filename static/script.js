const input = document.getElementById("text");
const keyboard = document.getElementById("keyboard");

async function fetchSuggestions(text) {
    try {
        const res = await fetch("/suggest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                top_k: 5
            })
        });

        const data = await res.json();
        renderKeyboard(data.suggestions);

    } catch (err) {
        console.error("API error:", err);
    }
}

function renderKeyboard(suggestions) {
    keyboard.innerHTML = "";

    suggestions.forEach(s => {
        const key = document.createElement("div");
        key.className = "key";

        key.innerHTML = `
            ${s.word}
            <span class="confidence">${(s.confidence * 100).toFixed(1)}%</span>
        `;

        key.onclick = () => {
            input.value += " " + s.word;
            fetchSuggestions(input.value);
        };

        keyboard.appendChild(key);
    });
}

input.addEventListener("input", e => {
    const text = e.target.value.trim();
    if (text.length > 0) {
        fetchSuggestions(text);
    } else {
        keyboard.innerHTML = "";
    }
});
