document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$",  right: "$$",  display: true },
      { left: "$",   right: "$",   display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
  })
})


// document.addEventListener("DOMContentLoaded", function() {
//   renderMathInElement(document.body, {
//     // customised options
//     // • auto-render specific keys, e.g.:
//     delimiters: [
//         {left: '$$', right: '$$', display: true},
//         {left: '$', right: '$', display: false},
//         {left: '\\(', right: '\\)', display: false},
//         {left: '\\[', right: '\\]', display: true}
//     ],
//     // • rendering keys, e.g.:
//     throwOnError : false
//   })
// })
