"use strict";
(() => {
  const data = window.LANG_DATA;
  function setLanguage(lang) {
    if (!Object.prototype.hasOwnProperty.call(data, lang)) lang = "en";
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const value = data[lang][node.dataset.i18n];
      if (typeof value === "string") node.textContent = value;
    });
    document.documentElement.lang = lang;
    document.title = lang === "tr"
      ? "Yağlı Tohum Sınıflandırması | TÜBİTAK 2209-A"
      : "Edible Oil Seed Classification | TÜBİTAK 2209-A";
    document.querySelectorAll("[data-lang]").forEach((node) => {
      if (node.dataset.lang === lang) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
      node.href = "?lang=" + node.dataset.lang + window.location.hash;
    });
    document.querySelectorAll("figure img").forEach((img) => {
      const accuracy = img.src.includes("accuracy");
      img.alt = lang === "tr"
        ? (accuracy ? "Özgün proje doğruluk karşılaştırması" : "Özgün proje makro F1 karşılaştırması")
        : (accuracy ? "Original-project accuracy comparison" : "Original-project macro F1 comparison");
    });
  }
  function fromUrl() {
    setLanguage(new URLSearchParams(window.location.search).get("lang") || "en");
  }
  document.querySelectorAll("[data-lang]").forEach((node) => {
    node.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const url = new URL(window.location.href);
      url.searchParams.set("lang", node.dataset.lang);
      window.history.pushState({}, "", url);
      setLanguage(node.dataset.lang);
    });
  });
  window.addEventListener("popstate", fromUrl);
  fromUrl();
  // Supply an official, unmodified local asset. Never draw or substitute a logo.
  const logo = new Image();
  logo.alt = "TÜBİTAK";
  logo.onload = () => {
    const slot = document.getElementById("logo-slot");
    slot.replaceChildren(logo);
    slot.classList.add("has-logo");
  };
  logo.onerror = () => {
    if (logo.src.endsWith(".png")) logo.src = "assets/tubitak-logo.svg";
  };
  logo.src = "assets/tubitak-logo.png";
})();
