document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    macros: {
      "\\bch": "\\mathrm{BCH}",
      "\\snr": "\\mathrm{SNR}",
      "\\Enc": "\\mathrm{Enc}",
      "\\Dec": "\\mathrm{Dec}",
    },
    throwOnError: false,
  });
});
