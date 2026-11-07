# Deploy de fraw.fabrictd.com (Netlify)

Sitio 100 % estático. Todo lo que se publica está en `web/`. La configuración vive en
`netlify.toml` en la **raíz** del repo:

```toml
[build]
  base    = "web"              # Netlify entra en web/ antes de construir
  publish = "."                # relativo a base ⇒ publica web/
  command = "npm run build:css"
```

`web/styles.css` ya está compilado y commiteado, así que el sitio funciona aunque el build
no corra. El comando sólo lo regenera para que nunca quede desfasado del HTML.

## Estructura

| Archivo | Para qué |
|---|---|
| `index.html`, `404.html`, `legal/*.html` | Páginas |
| `styles.css` | Tailwind compilado y minificado (commiteado) |
| `src/input.css`, `tailwind.config.js`, `package.json` | Fuentes de Tailwind |
| `assets/app.js` | Único JS: toggle ES/EN y cierre del menú móvil |
| `assets/favicon.svg`, `assets/og.png` | Isotipo e imagen para redes (1200×630) |
| `_redirects` | Atajos `/download`, `/github`, `/spec`, `/releases`, `/discord`, `/issues` |
| `robots.txt`, `sitemap.xml` | SEO |
| `scripts/check-links.mjs` | Verifica que cada `href`/`src` local exista |

### Trabajo local

```bash
cd web
npm install
npm run build:css      # regenera styles.css (commitearlo después)
npm run check:links    # verifica enlaces internos y anclas
npm run serve          # http://localhost:4173 (no aplica _redirects ni headers)
```

> Caché: `/assets/*` y `/styles.css` se sirven con caché de 1 año. El HTML los referencia
> con `?v=1.0.0`; **si se cambian, hay que subir ese número en todos los HTML**.

---

## (a) Repositorio de GitHub

El repo ya existe y es público: <https://github.com/FabriCTD/fabriraw>. Basta con que
`netlify.toml` y `web/` estén en la rama `main`.

**Si el repo fuera privado:** Netlify accede a través de su GitHub App.

1. Al importar, Netlify pide instalar/autorizar la app **Netlify** en la cuenta `FabriCTD`.
2. En GitHub → *Settings → Applications → Installed GitHub Apps → Netlify → Configure*,
   en *Repository access* elegir **Only select repositories** y agregar `fabriraw`
   (mínimo privilegio; no hace falta dar acceso a todos los repos).
3. La app necesita lectura de contenido y permiso para escribir *commit statuses /
   checks* y comentarios en PRs (para los deploy previews). Si después se quita el
   acceso al repo, los deploys dejan de dispararse.

## (b) Crear el sitio en Netlify

1. <https://app.netlify.com> → **Add new site → Import an existing project**.
2. **Deploy with GitHub** → autorizar → elegir `FabriCTD/fabriraw`.
3. Netlify lee `netlify.toml` y completa solo los campos. Verificar:
   - **Branch to deploy:** `main`
   - **Base directory:** `web`
   - **Build command:** `npm run build:css`
   - **Publish directory:** `web` (en la UI se muestra ya resuelto respecto de base)
4. **Deploy**. Queda publicado en `https://<nombre>.netlify.app`.
5. (Opcional) *Site configuration → Change site name* → p. ej. `fraw-fabrictd`.
6. Comprobar en el deploy: `/` carga, `/no-existe` muestra el 404 de la marca,
   `/spec` lleva a `/#spec`, y los headers (DevTools → Network → documento) incluyen
   `Content-Security-Policy` y `Strict-Transport-Security`.

Netlify sirve `404.html` automáticamente (con estado 404) para cualquier ruta inexistente,
por eso `_redirects` **no** tiene una regla `/* /404.html 404`.

## (c) Dominio personalizado `fraw.fabrictd.com`

1. Netlify → *Domain management → Add a domain* → `fraw.fabrictd.com` → **Verify**
   → *Add domain* (como el dominio raíz no está en Netlify DNS, ofrece configuración externa).
2. En el panel DNS donde esté `fabrictd.com` (Cloudflare u otro), crear:

   | Tipo | Nombre | Destino | TTL |
   |---|---|---|---|
   | `CNAME` | `fraw` | `<nombre>.netlify.app` | Auto |

   - **Cloudflare:** dejar el registro en **DNS only (nube gris)**. Con el proxy activo,
     Netlify no puede emitir el certificado y se duplica la caché.
   - No crear registros `A` para `fraw`; el CNAME alcanza.
3. Esperar la propagación (minutos, a veces horas). Comprobar con
   `nslookup fraw.fabrictd.com` → debe resolver a `*.netlify.app`.
4. **HTTPS:** Netlify emite y renueva automáticamente un certificado **Let's Encrypt**
   cuando el DNS apunta bien (*Domain management → HTTPS → Verify DNS configuration*).
   Activar **Force HTTPS** si no está activo.
5. Definir `fraw.fabrictd.com` como *Primary domain* para que `*.netlify.app` redirija ahí.

`netlify.toml` envía `Strict-Transport-Security: max-age=63072000; includeSubDomains`.
Sólo afecta a `fraw.fabrictd.com` y sus subdominios, no a `fabrictd.com`.

## (d) Publicar un release (para que funcione «Descargar»)

Todos los botones de descarga apuntan a:

```
https://github.com/FabriCTD/fabriraw/releases/latest/download/CTD-Photo-Studio-Portable-win64.zip
```

GitHub resuelve `latest` al último release **no draft y no pre-release**. Para que el
enlace funcione:

1. Actualizar la versión del proyecto y commitear.
2. Crear y subir un tag semántico:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. El workflow de GitHub Actions (disparado por tags `v*.*.*`) construye el portable de
   Windows y crea el release adjuntando el asset.
4. El nombre del asset debe ser **exactamente** `CTD-Photo-Studio-Portable-win64.zip`
   (mismo nombre en todas las versiones, sin número de versión). Si cambia, el enlace
   `latest/download/...` da 404.
5. Verificar: abrir `https://fraw.fabrictd.com/download` → debe bajar el ZIP.
6. Si el release se marcó como *pre-release*, `latest` lo ignora: desmarcarlo o publicar
   uno estable.

Mientras no exista ningún release, el enlace devuelve 404 de GitHub; la web lo avisa en la
sección Descargas.

## Pendientes de FabriCTD

- [ ] Configurar el CNAME `fraw` en el DNS de `fabrictd.com`.
- [ ] Definir la jurisdicción / ley aplicable en `legal/terminos.html`
      (buscar `TODO(FabriCTD)` en `legal/*.html`).
- [ ] Revisar los textos legales con un profesional (son plantillas informativas).
- [ ] Publicar el primer release con el asset `CTD-Photo-Studio-Portable-win64.zip`.
