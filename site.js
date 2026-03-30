window.document.addEventListener("DOMContentLoaded", () => {
  const interactiveCards = [
    ...window.document.querySelectorAll(".lead-feature, .secondary-feature, .quarto-post")
  ];

  interactiveCards.forEach((card) => {
    const primaryLink =
      card.querySelector(".listing-title a, h2 a, h3 a, .text-link, .metadata a");

    if (!primaryLink) {
      return;
    }

    card.classList.add("card-clickable");
    card.tabIndex = 0;
    card.setAttribute("role", "link");

    const labelSource = primaryLink.textContent?.trim();
    if (labelSource) {
      card.setAttribute("aria-label", labelSource);
    }

    card.addEventListener("click", (event) => {
      if (event.target instanceof Element && event.target.closest("a, button")) {
        return;
      }

      primaryLink.click();
    });

    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        primaryLink.click();
      }
    });
  });
});
