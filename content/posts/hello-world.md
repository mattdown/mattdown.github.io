---
title: "Hello World"
date: 2026-06-02
tags: ["meta"]
draft: true
---

This is a placeholder post. Replace it with something real.

You can embed a Unity WebGL build anywhere in a post using the `unity-webgl` shortcode:

```
{{</* unity-webgl
  buildPath="/games/my-game/Build"
  buildName="MyGame"
  title="My Game"
  width="960"
  height="600"
*/>}}
```

Place the Unity build files under `static/games/my-game/Build/` and they'll be served at `/games/my-game/Build/`.

MathJax is available per-post — add `mathjax: true` to the front matter to enable it. Then write inline math like $E = mc^2$ or display math:

$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
