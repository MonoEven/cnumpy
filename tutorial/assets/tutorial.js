/* cnumpy quickstart tutorial — shared behavior.
   1. Lightweight AutoHotkey v2 syntax highlighting for <pre data-lang="ahk">.
   2. TOC scroll-spy: marks the chapter currently in view.
   No external dependencies. */

(function () {
  "use strict";

  /* ---------- syntax highlighting ---------- */

  var AHK_KEYWORDS = new Set([
    "if", "else", "return", "try", "catch", "finally", "throw", "loop",
    "for", "in", "while", "break", "continue", "static", "class", "extends",
    "global", "local", "unset", "true", "false", "and", "or", "not", "is",
    "IsSet", "as", "until", "switch", "case", "goto"
  ]);

  var AHK_DIRECTIVE = /^#\w+/;

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* Tokenize one line of AHK source into highlighted HTML. */
  function highlightLine(line) {
    var out = "";
    var i = 0;
    var n = line.length;

    // Whole-line comment (allow leading spaces).
    var trimmed = line.replace(/^\s*/, "");
    var lead = line.slice(0, n - trimmed.length);
    if (trimmed.startsWith(";")) {
      return esc(lead) + '<span class="tok-c">' + esc(trimmed) + "</span>";
    }
    if (AHK_DIRECTIVE.test(trimmed)) {
      var m = trimmed.match(/^#\w+/)[0];
      return esc(lead) + '<span class="tok-k">' + esc(m) + "</span>" +
        esc(trimmed.slice(m.length));
    }

    while (i < n) {
      var ch = line[i];

      // Trailing comment: space + ;
      if (ch === ";" && (i === 0 || line[i - 1] === " " || line[i - 1] === "\t")) {
        out += '<span class="tok-c">' + esc(line.slice(i)) + "</span>";
        break;
      }

      // Strings
      if (ch === '"') {
        var j = i + 1;
        while (j < n && line[j] !== '"') j++;
        out += '<span class="tok-s">' + esc(line.slice(i, j + 1)) + "</span>";
        i = j + 1;
        continue;
      }

      // Numbers
      if (/[0-9]/.test(ch) && (i === 0 || !/[\w.]/.test(line[i - 1]))) {
        var k = i;
        while (k < n && /[0-9a-fA-FxX.eE_+-]/.test(line[k])) {
          if ((line[k] === "+" || line[k] === "-") &&
              !/[eE]/.test(line[k - 1])) break;
          k++;
        }
        out += '<span class="tok-n">' + esc(line.slice(i, k)) + "</span>";
        i = k;
        continue;
      }

      // Identifiers
      if (/[A-Za-z_]/.test(ch)) {
        var w = i;
        while (w < n && /[\w]/.test(line[w])) w++;
        var word = line.slice(i, w);
        var next = line.slice(w).match(/^\s*[(.]/);
        if (AHK_KEYWORDS.has(word)) {
          out += '<span class="tok-k">' + esc(word) + "</span>";
        } else if (word === "Numpy" || word === "NdArray" || word === "A_ScriptDir" ||
                   word === "MsgBox" || word === "FileAppend" || word === "Error" ||
                   word === "TypeError" || word === "ValueError" || word === "Format" ||
                   word === "ExitApp" || word === "Buffer" || word === "Map") {
          out += '<span class="tok-cls">' + esc(word) + "</span>";
        } else if (next && next[0].trim() === "(") {
          out += '<span class="tok-f">' + esc(word) + "</span>";
        } else {
          out += esc(word);
        }
        i = w;
        continue;
      }

      // Operators
      if (/[:=<>!+\-*\/.&|^~?]/.test(ch)) {
        out += '<span class="tok-op">' + esc(ch) + "</span>";
        i++;
        continue;
      }

      out += esc(ch);
      i++;
    }
    return out;
  }

  function highlightAll() {
    var blocks = document.querySelectorAll('pre[data-lang="ahk"] > code');
    blocks.forEach(function (codeEl) {
      var lines = codeEl.textContent.split("\n");
      codeEl.innerHTML = lines.map(highlightLine).join("\n");
    });
  }

  /* ---------- TOC scroll-spy ---------- */

  function initScrollSpy() {
    var links = Array.prototype.slice.call(
      document.querySelectorAll("nav.toc a[href^='#']"));
    if (!links.length || !("IntersectionObserver" in window)) return;

    var byId = {};
    links.forEach(function (a) {
      byId[a.getAttribute("href").slice(1)] = a;
    });

    var current = null;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var link = byId[entry.target.id];
        if (!link) return;
        if (current) current.classList.remove("active");
        link.classList.add("active");
        current = link;
      });
    }, { rootMargin: "-10% 0px -75% 0px", threshold: 0 });

    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

  /* ---------- language switch: preserve the current anchor ---------- */

  function initLangSwitch() {
    var links = document.querySelectorAll(".lang-switch a[data-target]");
    links.forEach(function (a) {
      a.addEventListener("click", function (event) {
        event.preventDefault();
        window.location.href = a.getAttribute("data-target") + window.location.hash;
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      highlightAll();
      initScrollSpy();
      initLangSwitch();
    });
  } else {
    highlightAll();
    initScrollSpy();
    initLangSwitch();
  }
})();
