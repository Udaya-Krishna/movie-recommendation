async function getRecommendations() {
  const movie = document.getElementById("movieInput").value;
  const url = `http://127.0.0.1:8000/recommend?movie=${encodeURIComponent(movie)}`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Movie not found");

    const data = await response.json();
    const recList = document.getElementById("recommendations");
    recList.innerHTML = "";

    // Heading
    let heading = document.createElement("h3");
    heading.textContent = `Recommendations for "${data.movie}":`;
    recList.appendChild(heading);

    // Recommended movies
    data.recommendations.forEach(movie => {
      let li = document.createElement("li");

      let img = document.createElement("img");
      img.src = movie.poster || "https://via.placeholder.com/100x150?text=No+Image";
      img.alt = movie.title;
      img.width = 120;

      let title = document.createElement("p");
      title.textContent = movie.title;

      li.appendChild(img);
      li.appendChild(title);
      recList.appendChild(li);
    });

  } catch (error) {
    alert("Error: " + error.message);
  }
}

// 🔹 Autocomplete Suggestions
async function getSuggestions() {
  const query = document.getElementById("movieInput").value;
  if (query.length < 2) return; // wait for at least 2 letters

  const url = `http://127.0.0.1:8000/suggest?q=${encodeURIComponent(query)}`;
  const response = await fetch(url);
  const data = await response.json();

  const suggestionsBox = document.getElementById("suggestions");
  suggestionsBox.innerHTML = "";

  data.suggestions.forEach(s => {
    let option = document.createElement("div");
    option.textContent = s;
    option.classList.add("suggestion-item");

    option.onclick = () => {
      document.getElementById("movieInput").value = s;
      suggestionsBox.innerHTML = "";
    };

    suggestionsBox.appendChild(option);
  });
}
