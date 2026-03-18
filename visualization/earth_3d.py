"""
Space Debris AI — 3D Earth Visualization
Generates a self-contained HTML/Three.js interactive globe with orbiting debris.
"""

from typing import List, Dict


def generate_earth_html(debris_data: List[Dict], width: int = 800, height: int = 700) -> str:
    """
    Generate a self-contained HTML page with a Three.js Earth globe
    and orbiting debris objects.

    Args:
        debris_data: List of debris dicts with lat, lon, altitude, risk_level, name, etc.
        width: Canvas width
        height: Canvas height

    Returns:
        Complete HTML string
    """
    # Convert debris data to JS array
    debris_js = _debris_to_js(debris_data)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #000010;
        overflow: hidden;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    #container {{
        width: {width}px;
        height: {height}px;
        position: relative;
    }}
    canvas {{
        display: block;
    }}
    #tooltip {{
        position: absolute;
        background: rgba(10, 15, 30, 0.92);
        border: 1px solid rgba(0, 180, 255, 0.4);
        border-radius: 10px;
        padding: 14px 18px;
        color: #e0e8ff;
        font-size: 13px;
        pointer-events: none;
        display: none;
        z-index: 100;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 30px rgba(0, 120, 255, 0.15);
        min-width: 200px;
        line-height: 1.6;
    }}
    #tooltip .title {{
        font-weight: 700;
        font-size: 14px;
        color: #00d4ff;
        margin-bottom: 6px;
    }}
    #tooltip .risk {{
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        display: inline-block;
    }}
    #stats-panel {{
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(10, 15, 30, 0.85);
        border: 1px solid rgba(0, 180, 255, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        color: #c0d0ff;
        font-size: 12px;
        z-index: 50;
        backdrop-filter: blur(6px);
        line-height: 1.8;
    }}
    #stats-panel h3 {{
        color: #00d4ff;
        font-size: 13px;
        margin-bottom: 6px;
        letter-spacing: 1px;
    }}
    .legend-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }}
</style>
</head>
<body>
<div id="container">
    <div id="tooltip"></div>
    <div id="stats-panel">
        <h3>🛰️ ORBITAL DEBRIS MONITOR</h3>
        <div id="stats-content"></div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {{
    // ─── Debris Data ────────────────────────────────────
    const debrisData = {debris_js};

    // ─── Scene Setup ────────────────────────────────────
    const container = document.getElementById('container');
    const tooltip = document.getElementById('tooltip');
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(50, {width}/{height}, 0.1, 2000);
    camera.position.set(0, 0, 4.5);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize({width}, {height});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000010);
    container.prepend(renderer.domElement);

    // ─── Starfield ──────────────────────────────────────
    const starsGeo = new THREE.BufferGeometry();
    const starPositions = new Float32Array(3000 * 3);
    const starColors = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) {{
        const r = 50 + Math.random() * 150;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        starPositions[i*3] = r * Math.sin(phi) * Math.cos(theta);
        starPositions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
        starPositions[i*3+2] = r * Math.cos(phi);
        const brightness = 0.4 + Math.random() * 0.6;
        starColors[i*3] = brightness;
        starColors[i*3+1] = brightness;
        starColors[i*3+2] = brightness + Math.random() * 0.2;
    }}
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starsGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const starsMat = new THREE.PointsMaterial({{ size: 0.15, vertexColors: true, transparent: true, opacity: 0.8 }});
    scene.add(new THREE.Points(starsGeo, starsMat));

    // ─── Lighting ───────────────────────────────────────
    const ambient = new THREE.AmbientLight(0x334466, 0.6);
    scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(5, 3, 5);
    scene.add(sun);
    const rim = new THREE.DirectionalLight(0x4488ff, 0.3);
    rim.position.set(-3, -1, -3);
    scene.add(rim);

    // ─── Earth ──────────────────────────────────────────
    const earthRadius = 1.0;
    const earthGeo = new THREE.SphereGeometry(earthRadius, 64, 64);

    // Procedural Earth-like colors using shader
    const earthMat = new THREE.ShaderMaterial({{
        uniforms: {{
            time: {{ value: 0 }},
            sunDir: {{ value: new THREE.Vector3(1, 0.5, 1).normalize() }}
        }},
        vertexShader: `
            varying vec3 vNormal;
            varying vec3 vPosition;
            varying vec2 vUv;
            void main() {{
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }}
        `,
        fragmentShader: `
            uniform vec3 sunDir;
            uniform float time;
            varying vec3 vNormal;
            varying vec3 vPosition;
            varying vec2 vUv;

            // Simple hash-based noise
            float hash(vec2 p) {{
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }}
            float noise(vec2 p) {{
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }}

            void main() {{
                float n = noise(vUv * 8.0) * 0.5 + noise(vUv * 16.0) * 0.25 + noise(vUv * 32.0) * 0.125;

                // Ocean & land colors
                vec3 ocean = vec3(0.02, 0.08, 0.28);
                vec3 shallow = vec3(0.04, 0.18, 0.42);
                vec3 land = vec3(0.12, 0.35, 0.08);
                vec3 mountain = vec3(0.35, 0.28, 0.15);
                vec3 ice = vec3(0.85, 0.9, 0.95);

                float landMask = smoothstep(0.42, 0.52, n);
                vec3 terrainColor = mix(ocean, shallow, smoothstep(0.3, 0.42, n));
                terrainColor = mix(terrainColor, land, landMask);
                terrainColor = mix(terrainColor, mountain, smoothstep(0.65, 0.8, n));

                // Polar ice
                float polar = abs(vUv.y - 0.5) * 2.0;
                terrainColor = mix(terrainColor, ice, smoothstep(0.75, 0.95, polar));

                // Diffuse lighting
                float diff = max(dot(vNormal, sunDir), 0.0);
                diff = diff * 0.7 + 0.3;

                // Atmosphere rim
                vec3 viewDir = normalize(-vPosition);
                float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
                vec3 atmosColor = vec3(0.3, 0.6, 1.0);
                vec3 finalColor = terrainColor * diff + atmosColor * pow(rim, 3.0) * 0.5;

                gl_FragColor = vec4(finalColor, 1.0);
            }}
        `
    }});

    const earth = new THREE.Mesh(earthGeo, earthMat);
    scene.add(earth);

    // Atmosphere glow
    const atmosGeo = new THREE.SphereGeometry(earthRadius * 1.02, 64, 64);
    const atmosMat = new THREE.ShaderMaterial({{
        uniforms: {{}},
        vertexShader: `
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {{
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }}
        `,
        fragmentShader: `
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {{
                vec3 viewDir = normalize(-vPosition);
                float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
                vec3 color = vec3(0.3, 0.6, 1.0);
                float alpha = pow(rim, 4.0) * 0.6;
                gl_FragColor = vec4(color, alpha);
            }}
        `,
        transparent: true,
        side: THREE.FrontSide,
        depthWrite: false,
    }});
    scene.add(new THREE.Mesh(atmosGeo, atmosMat));

    // ─── Orbital Rings ──────────────────────────────────
    const ringAltitudes = [0.3, 0.5, 0.8, 1.1];
    ringAltitudes.forEach(alt => {{
        const ringGeo = new THREE.RingGeometry(earthRadius + alt - 0.005, earthRadius + alt + 0.005, 128);
        const ringMat = new THREE.MeshBasicMaterial({{
            color: 0x1a4a7a,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
        }});
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        scene.add(ring);
    }});

    // ─── Debris Objects ─────────────────────────────────
    const riskColors = {{
        'High': new THREE.Color(0xff2244),
        'Medium': new THREE.Color(0xffaa00),
        'Low': new THREE.Color(0x22dd66),
    }};

    const debrisMeshes = [];
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    debrisData.forEach((d, idx) => {{
        const color = riskColors[d.risk_level] || riskColors['Low'];
        const size = 0.012 + (d.size || 5) * 0.0005;

        // Debris sphere
        const geo = new THREE.SphereGeometry(size, 8, 8);
        const mat = new THREE.MeshBasicMaterial({{ color: color }});
        const mesh = new THREE.Mesh(geo, mat);

        // Glow
        const glowGeo = new THREE.SphereGeometry(size * 2.2, 8, 8);
        const glowMat = new THREE.MeshBasicMaterial({{
            color: color,
            transparent: true,
            opacity: 0.15,
        }});
        const glow = new THREE.Mesh(glowGeo, glowMat);
        mesh.add(glow);

        // Orbit parameters
        const orbitRadius = earthRadius + (d.altitude / 2000) * 1.2;
        const inclination = (d.inclination || 45) * Math.PI / 180;
        const phase = d.orbit_phase || Math.random() * Math.PI * 2;
        const speed = d.angular_speed || 0.005 + Math.random() * 0.015;

        mesh.userData = {{
            ...d,
            orbitRadius,
            inclination,
            phase,
            speed,
            idx,
        }};

        scene.add(mesh);
        debrisMeshes.push(mesh);
    }});

    // ─── Orbit Controls (simplified) ────────────────────
    let isDragging = false;
    let prevMouse = {{ x: 0, y: 0 }};
    let rotation = {{ x: 0.3, y: 0 }};
    let targetRotation = {{ x: 0.3, y: 0 }};
    let zoom = 4.5;
    let targetZoom = 4.5;

    container.addEventListener('mousedown', e => {{
        isDragging = true;
        prevMouse = {{ x: e.clientX, y: e.clientY }};
    }});
    container.addEventListener('mousemove', e => {{
        if (isDragging) {{
            const dx = e.clientX - prevMouse.x;
            const dy = e.clientY - prevMouse.y;
            targetRotation.y += dx * 0.005;
            targetRotation.x += dy * 0.005;
            targetRotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, targetRotation.x));
            prevMouse = {{ x: e.clientX, y: e.clientY }};
        }}

        // Tooltip on hover
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(debrisMeshes);

        if (intersects.length > 0) {{
            const d = intersects[0].object.userData;
            const riskColor = d.risk_level === 'High' ? '#ff4444' :
                              d.risk_level === 'Medium' ? '#ffaa00' : '#44dd44';
            tooltip.innerHTML = `
                <div class="title">${{d.name || d.id}}</div>
                <div>📍 Lat: ${{d.lat}}° &nbsp; Lon: ${{d.lon}}°</div>
                <div>📏 Altitude: ${{d.altitude}} km</div>
                <div>⚡ Velocity: ${{d.velocity}} km/s</div>
                <div>📐 Size: ${{d.size}} cm</div>
                <div>🛤️ Orbit: ${{d.orbit_type}}</div>
                <div>Risk: <span class="risk" style="background:${{riskColor}}33;color:${{riskColor}}">${{d.risk_level}}</span></div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
            tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
        }} else {{
            tooltip.style.display = 'none';
        }}
    }});
    container.addEventListener('mouseup', () => {{ isDragging = false; }});
    container.addEventListener('mouseleave', () => {{ isDragging = false; tooltip.style.display = 'none'; }});
    container.addEventListener('wheel', e => {{
        e.preventDefault();
        targetZoom += e.deltaY * 0.003;
        targetZoom = Math.max(2, Math.min(10, targetZoom));
    }}, {{ passive: false }});

    // ─── Stats Panel ────────────────────────────────────
    const riskDist = {{ High: 0, Medium: 0, Low: 0 }};
    debrisData.forEach(d => riskDist[d.risk_level]++);
    document.getElementById('stats-content').innerHTML = `
        <div>Total Objects: <strong>${{debrisData.length}}</strong></div>
        <div><span class="legend-dot" style="background:#ff2244"></span>High Risk: <strong>${{riskDist.High}}</strong></div>
        <div><span class="legend-dot" style="background:#ffaa00"></span>Medium Risk: <strong>${{riskDist.Medium}}</strong></div>
        <div><span class="legend-dot" style="background:#22dd66"></span>Low Risk: <strong>${{riskDist.Low}}</strong></div>
    `;

    // ─── Animation Loop ─────────────────────────────────
    let time = 0;
    function animate() {{
        requestAnimationFrame(animate);
        time += 0.005;

        // Smooth controls
        rotation.x += (targetRotation.x - rotation.x) * 0.08;
        rotation.y += (targetRotation.y - rotation.y) * 0.08;
        zoom += (targetZoom - zoom) * 0.08;

        // Auto-rotate
        targetRotation.y += 0.001;

        camera.position.x = zoom * Math.sin(rotation.y) * Math.cos(rotation.x);
        camera.position.y = zoom * Math.sin(rotation.x);
        camera.position.z = zoom * Math.cos(rotation.y) * Math.cos(rotation.x);
        camera.lookAt(0, 0, 0);

        // Update Earth shader
        earthMat.uniforms.time.value = time;

        // Update debris positions (orbital motion)
        debrisMeshes.forEach(mesh => {{
            const u = mesh.userData;
            const angle = u.phase + time * u.speed * 50;
            const r = u.orbitRadius;
            const inc = u.inclination;

            mesh.position.x = r * Math.cos(angle);
            mesh.position.y = r * Math.sin(angle) * Math.sin(inc);
            mesh.position.z = r * Math.sin(angle) * Math.cos(inc);
        }});

        renderer.render(scene, camera);
    }}
    animate();
}})();
</script>
</body>
</html>
"""
    return html


def _debris_to_js(debris_data: List[Dict]) -> str:
    """Convert Python debris list to JavaScript array literal."""
    import json
    return json.dumps(debris_data)
