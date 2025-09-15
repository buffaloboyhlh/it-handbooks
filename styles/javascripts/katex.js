document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },   // 支持 $$...$$
      { left: "\\[", right: "\\]", display: true }, // 支持 \[...\]
      { left: "$", right: "$", display: false },    // 支持 $...$
      { left: "\\(", right: "\\)", display: false } // 支持 \(...\)
    ],
    throwOnError: false
  });
});