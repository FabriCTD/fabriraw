/* fraw.fabrictd.com — único JS del sitio: toggle ES/EN (persistido) y cierre del menú móvil.
   El sitio funciona completo sin JS (en español; el menú móvil es un <details> nativo).
   Sin cookies, sin trackers, sin peticiones a terceros. */
(function () {
  'use strict';
  var KEY = 'fraw-lang';
  var root = document.documentElement;

  function load() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function save(v) {
    try { window.localStorage.setItem(KEY, v); } catch (e) { /* almacenamiento bloqueado */ }
  }

  function apply(lang) {
    var en = lang === 'en';
    root.setAttribute('data-lang', en ? 'en' : 'es');
    root.setAttribute('lang', en ? 'en' : 'es');
    var t = root.getAttribute(en ? 'data-title-en' : 'data-title-es');
    if (t) document.title = t;
    var btns = document.querySelectorAll('[data-lang-toggle]');
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute('aria-pressed', en ? 'true' : 'false');
    }
  }

  var saved = load();
  apply(saved === 'en' ? 'en' : 'es');

  // El botón de idioma sólo funciona con JS: se muestra aquí.
  var js = document.querySelectorAll('[data-js-only]');
  for (var j = 0; j < js.length; j++) js[j].hidden = false;

  document.addEventListener('click', function (ev) {
    var t = ev.target;
    if (!t || !t.closest) return;
    var btn = t.closest('[data-lang-toggle]');
    if (btn) {
      var next = root.getAttribute('data-lang') === 'en' ? 'es' : 'en';
      apply(next);
      save(next);
      return;
    }
    var link = t.closest('#mobile-menu a');
    if (link) {
      var d = document.getElementById('mobile-menu');
      if (d) d.open = false;
    }
  });

  document.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Escape') return;
    var d = document.getElementById('mobile-menu');
    if (d && d.open) {
      d.open = false;
      var s = d.querySelector('summary');
      if (s) s.focus();
    }
  });
})();
