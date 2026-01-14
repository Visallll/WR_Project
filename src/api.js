export async function getSuggestions(text) {
  const res = await fetch("/suggest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, top_k: 5 })
  });
  return res.json();
}
