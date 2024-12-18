document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    macros: {
      "\\transpose": "\\mathsf{T}",
      "\\bch": "\\mathrm{BCH}",
      "\\snr": "\\mathrm{SNR}",
      "\\Enc": "\\mathrm{Enc}",
      "\\Dec": "\\mathrm{Dec}",
      "\\Hb": "\\mathrm{H}_\\mathrm{b}",
    },
    throwOnError: false,
  });
});
