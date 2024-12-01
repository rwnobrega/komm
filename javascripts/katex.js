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
      "\\dH": "\\mathrm{d}_\\mathrm{H}",
      "\\wH": "\\mathrm{w}_\\mathrm{H}",
    },
    throwOnError: false,
  });
});
