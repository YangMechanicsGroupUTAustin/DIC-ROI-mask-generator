/* ══════════════════════════════════════════════════
   DIC Mask Generator — Website Scripts
   ══════════════════════════════════════════════════ */

(function () {
  "use strict";

  /* ── Scroll-triggered animations ── */
  const animatedElements = document.querySelectorAll("[data-animate]");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const delay = parseInt(el.dataset.delay || "0", 10);
        setTimeout(() => el.classList.add("is-visible"), delay);
        observer.unobserve(el);
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  animatedElements.forEach((el) => observer.observe(el));

  /* ── Navbar scroll effect ── */
  const navbar = document.getElementById("navbar");
  let lastScroll = 0;

  function onScroll() {
    const y = window.scrollY;
    if (y > 60) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
    lastScroll = y;
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ── Active nav link ── */
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-links a[href^='#']");

  function updateActiveLink() {
    const scrollY = window.scrollY + 120;

    sections.forEach((section) => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      const id = section.getAttribute("id");

      if (scrollY >= top && scrollY < top + height) {
        navLinks.forEach((link) => {
          link.classList.remove("active");
          if (link.getAttribute("href") === `#${id}`) {
            link.classList.add("active");
          }
        });
      }
    });
  }

  window.addEventListener("scroll", updateActiveLink, { passive: true });
  updateActiveLink();

  /* ── Mobile menu toggle ── */
  const navToggle = document.querySelector(".nav-toggle");
  const navLinksContainer = document.querySelector(".nav-links");

  if (navToggle && navLinksContainer) {
    navToggle.addEventListener("click", () => {
      navToggle.classList.toggle("active");
      navLinksContainer.classList.toggle("open");
    });

    // Close on link click
    navLinksContainer.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navToggle.classList.remove("active");
        navLinksContainer.classList.remove("open");
      });
    });

    // Close on outside click
    document.addEventListener("click", (e) => {
      if (!navToggle.contains(e.target) && !navLinksContainer.contains(e.target)) {
        navToggle.classList.remove("active");
        navLinksContainer.classList.remove("open");
      }
    });
  }

  /* ── Copy buttons ── */
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      let text;

      // BibTeX copy
      if (btn.id === "bibtex-copy") {
        const pre = document.getElementById("bibtex-content");
        text = pre ? pre.textContent : "";
      } else {
        // Code block copy
        const targetId = btn.dataset.target;
        const pre = document.getElementById(targetId);
        text = pre ? pre.textContent : "";
      }

      if (!text) return;

      navigator.clipboard.writeText(text).then(() => {
        const label = btn.querySelector("span");
        const originalText = label.textContent;
        btn.classList.add("copied");
        label.textContent = "Copied!";

        setTimeout(() => {
          btn.classList.remove("copied");
          label.textContent = originalText;
        }, 2000);
      }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand("copy");
          const label = btn.querySelector("span");
          const originalText = label.textContent;
          btn.classList.add("copied");
          label.textContent = "Copied!";
          setTimeout(() => {
            btn.classList.remove("copied");
            label.textContent = originalText;
          }, 2000);
        } catch (_) {
          // Silently fail
        }
        document.body.removeChild(textarea);
      });
    });
  });
})();
