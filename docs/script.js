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
    document.title = lang === "tr" ? "Yenilebilir Yağlı Tohumların Sınıflandırılması | Araştırma Projesi" : "Edible Oil Seed Classification | Research Project";
    document.querySelectorAll("[data-lang]").forEach((node) => {
      if (node.dataset.lang === lang) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
      node.href = "?lang=" + node.dataset.lang + window.location.hash;
    });
  }
  function fromUrl() { setLanguage(new URLSearchParams(window.location.search).get("lang") || "en"); }
  document.querySelectorAll("[data-lang]").forEach((node) => node.addEventListener("click", (event) => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const url = new URL(window.location.href);
    url.searchParams.set("lang", node.dataset.lang);
    window.history.pushState({}, "", url);
    setLanguage(node.dataset.lang);
  }));
  window.addEventListener("popstate", fromUrl);
  fromUrl();
})();
