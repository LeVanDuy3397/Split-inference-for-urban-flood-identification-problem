

        // ====== Khởi tạo bản đồ ======
        const map = L.map('map', { zoomControl: true, preferCanvas: true });
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        }).addTo(map);
        map.setView([16.0471, 108.2062], 6);

        let currentMarker = null;
        let pathLine = L.polyline([], { color: '#3b82f6', weight: 3, opacity: .8 }).addTo(map);

        const floodLayerGroup = L.layerGroup().addTo(map);
        const floodCircles = [];
        let filterMode = 'none';

        function applyFilter() {
            floodCircles.forEach(obj => {
                const { circle, maxLevel, baseColor, baseFill } = obj;
                const bucket = maxLevel === 0 ? '0' : (maxLevel <= 2 ? '1-2' : '3+');
                if (filterMode === 'none') {
                    circle.setStyle({ color: baseColor, fillColor: baseFill, opacity: 1, fillOpacity: .4 });
                } else if (filterMode === bucket) {
                    circle.setStyle({ color: baseColor, fillColor: baseFill, opacity: 1, fillOpacity: .5 });
                    circle.bringToFront();
                } else {
                    circle.setStyle({ color: '#6b7280', fillColor: '#6b7280', opacity: .5, fillOpacity: .15 });
                }
            });
        }
        function setFilter(mode) {
            filterMode = mode;
            document.querySelectorAll('.legend-item').forEach(el => el.classList.toggle('active', el.dataset.range === mode));
            applyFilter();
        }
        map.on('click', () => setFilter('none'));
        document.getElementById('btnClearFilter').addEventListener('click', () => setFilter('none'));
        document.querySelectorAll('.legend-item').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                const key = el.dataset.range;
                setFilter(filterMode === key ? 'none' : key);
            });
        });

        function addPulse(latLng) {
            const tpl = document.getElementById('pulse-template');
            const node = tpl.content.firstElementChild.cloneNode(true);
            const pane = map.getPanes().overlayPane;
            pane.appendChild(node);
            function updatePos() {
                const p = map.latLngToLayerPoint(latLng);
                node.style.transform = `translate(${p.x}px, ${p.y}px)`;
            }
            updatePos();
            const onZoom = () => requestAnimationFrame(updatePos);
            map.on('zoom viewreset move', onZoom);
            setTimeout(() => {
                map.off('zoom viewreset move', onZoom);
                node.remove();
            }, 1100);
        }

        function showPoint(lat, lng, timestamp, prediction) {
            if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
            const pred = prediction;
            const levels = (Array.isArray(pred) ? pred : [pred]).map(n => Number(n)).filter(n => Number.isInteger(n) && n >= 0);
            if (!levels.length) return;

            const uniqueDesc = [...new Set(levels)].sort((a, b) => b - a);
            const maxLevel = uniqueDesc[0];

            let text;
            if (uniqueDesc.length === 1) {
                text = `Thời gian: ${timestamp}<br>Dự đoán khu vực đang ngập ở mức ${maxLevel}`;
            } else {
                const others = uniqueDesc.slice(1).sort((a, b) => a - b);
                const othersText = others.map(l => `mức ${l}`).join(', ');
                text = `Thời gian: ${timestamp}<br>Dự đoán khu vực có thể đang ngập tới mức ${maxLevel} có vị trí ngập ${othersText}`;
            }

            const hasRed = uniqueDesc.some(l => l > 2);
            const baseColor = (maxLevel === 0) ? 'green' : (hasRed ? 'red' : 'orange');
            const baseFill = (maxLevel === 0) ? '#22c55e' : (hasRed ? '#ef4444' : '#ffcc00');

            const latLng = [lat, lng];
            const circle = L.circle(latLng, { color: baseColor, fillColor: baseFill, fillOpacity: 0.4, radius: 18, weight: 2 }).bindPopup(text);
            circle.addTo(floodLayerGroup);
            floodCircles.push({ circle, levelSet: new Set(uniqueDesc), maxLevel, baseColor, baseFill });
            applyFilter();

            circle.on('add', () => { addPulse(latLng); });

            if (!window.currentMarker) window.currentMarker = L.marker(latLng).addTo(map);
            else window.currentMarker.setLatLng(latLng);

            if (window.pathLine) window.pathLine.addLatLng(latLng);
            const currentZoom = map.getZoom();
            map.setView(latLng, Math.max(currentZoom, 17));
        }
        // ====== Helpers chung & Overlay ======
        function setAppBlocking(on, title = '', sub = '') {
            const overlay = document.getElementById('appOverlay');
            const t = document.getElementById('overlayTitle');
            const s = document.getElementById('overlaySub');
            const controls = document.querySelectorAll('button, input, select, textarea');
            controls.forEach(el => { el.disabled = !!on; });
            if (on) {
                t.textContent = title || 'Đang xử lý…';
                s.textContent = sub || 'Vui lòng đợi trong giây lát.';
                overlay.classList.add('show');
                overlay.setAttribute('aria-hidden', 'false');
            } else {
                overlay.classList.remove('show');
                overlay.setAttribute('aria-hidden', 'true');
            }
        }

        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const result = reader.result;
                    const idx = result.indexOf(',');
                    resolve(idx >= 0 ? result.slice(idx + 1) : result);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        function isSecureContextForGeo() {
            return window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
        }

        function formatTimestamp(d = new Date()) {
            const y = d.getFullYear();
            const m = String(d.getMonth() + 1).padStart(2, '0');
            const da = String(d.getDate()).padStart(2, '0');
            const h = String(d.getHours()).padStart(2, '0');
            const mi = String(d.getMinutes()).padStart(2, '0');
            const s = String(d.getSeconds()).padStart(2, '0');
            return `${y}-${m}-${da} ${h}:${mi}:${s}`;
        }

        // ====== Gửi ảnh ======
        const btnSendImage = document.getElementById('btnSendImage');
        const imageInput = document.getElementById('imageInput');
        const imgStatus = document.getElementById('imgStatus');

        btnSendImage.addEventListener('click', async () => {
            imgStatus.textContent = '';
            imgStatus.className = 'status';
            try {
                const file = imageInput.files && imageInput.files[0];
                if (!file) {
                    imgStatus.textContent = 'Vui lòng chọn ảnh trước.';
                    imgStatus.classList.add('err');
                    return;
                }
                setAppBlocking(true, 'AI đang dự đoán mức ngập…', 'Hệ thống đang phân tích ảnh của bạn.');
                const base64 = await fileToBase64(file);
                const res = await fetch('/api/send-image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_base64: base64, sent_at: Date.now() })
                });
                if (!res.ok) throw new Error('Gửi ảnh thất bại');
                const data = await res.json();
                if (data.ok) {
                    document.getElementById('overlayTitle').textContent = 'Đã dự đoán xong mức ngập tại khu vực';
                    document.getElementById('overlaySub').textContent = 'Nhấn "Gửi vị trí" để ghi nhận.';
                    setTimeout(() => setAppBlocking(false), 800);
                    imgStatus.textContent = 'Đã dự đoán xong mức ngập tại khu vực, nhấn Gửi vị trí để ghi nhận.';
                    imgStatus.classList.add('ok');
                } else {
                    throw new Error(data.error || 'Gửi ảnh thất bại');
                }
            } catch (e) {
                setAppBlocking(false);
                imgStatus.textContent = 'Lỗi: ' + e.message;
                imgStatus.classList.add('err');
            }
        });

        // ====== Gửi video (tách frame và gửi như ảnh) ======
        const btnSendVideo = document.getElementById('btnSendVideo');
        const videoInput = document.getElementById('videoInput');
        const videoStatus = document.getElementById('videoStatus');
        const videoProgressBar = document.getElementById('videoProgress');

        const fpsOut = 2;

        btnSendVideo.addEventListener('click', async () => {
            videoStatus.textContent = '';
            videoStatus.className = 'status';
            videoProgressBar.style.width = '0%';

            try {
                const file = videoInput.files && videoInput.files[0];
                if (!file) {
                    videoStatus.textContent = 'Vui lòng chọn video trước.';
                    videoStatus.classList.add('err');
                    return;
                }

                setAppBlocking(true, 'AI đang dự đoán mức ngập…', 'Đang trích xuất khung hình từ video của bạn.');

                const url = URL.createObjectURL(file);
                const video = document.createElement('video');
                video.src = url;
                video.muted = true;
                video.playsInline = true;
                await video.play().catch(() => {});
                await new Promise(r => {
                    if (video.readyState >= 2) r();
                    else video.addEventListener('loadeddata', () => r(), { once: true });
                });

                const duration = video.duration || 0;
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                const maxW = 640;
                const scale = Math.min(1, maxW / (video.videoWidth || maxW));
                canvas.width = Math.max(1, Math.round((video.videoWidth || maxW) * scale));
                canvas.height = Math.max(1, Math.round((video.videoHeight || (maxW * 9/16)) * scale));

                const totalFrames = Math.max(1, Math.floor(duration * fpsOut));
                let sent = 0;

                fetch('/api/send-video-meta', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: file.name, duration, fps: fpsOut, width: canvas.width, height: canvas.height })
                }).catch(()=>{});

                for (let i = 0; i < totalFrames; i++) {
                    const t = Math.min(duration, i / fpsOut);
                    await new Promise((resolve, reject) => {
                        const onSeeked = () => { cleanup(); resolve(); };
                        const onError = () => { cleanup(); reject(new Error('Không thể đọc khung hình video.')); };
                        const cleanup = () => {
                            video.removeEventListener('seeked', onSeeked);
                            video.removeEventListener('error', onError);
                        };
                        video.addEventListener('seeked', onSeeked, { once: true });
                        video.addEventListener('error', onError, { once: true });
                        try { video.currentTime = t; } catch(e) { cleanup(); reject(e); }
                    });

                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                    const base64 = dataUrl.split(',')[1];

                    const res = await fetch('/api/send-image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_base64: base64, sent_at: Date.now(), from_video: true, frame_index: i, fps: fpsOut })
                    });
                    if (!res.ok) throw new Error('Gửi khung hình thất bại');

                    sent++;
                    const pct = Math.round((sent / totalFrames) * 100);
                    videoProgressBar.style.width = pct + '%';
                    document.getElementById('overlaySub').textContent = `Đang gửi khung hình ${sent}/${totalFrames}…`;
                }

                document.getElementById('overlayTitle').textContent = 'Đã dự đoán xong mức ngập từ video';
                document.getElementById('overlaySub').textContent = 'Nhấn "Gửi vị trí" để ghi nhận khu vực.';
                setTimeout(() => setAppBlocking(false), 800);

                videoStatus.textContent = `Đã trích và gửi ${sent}/${totalFrames} khung hình.`;
                videoStatus.classList.add('ok');

                URL.revokeObjectURL(url);
            } catch (e) {
                setAppBlocking(false);
                videoStatus.textContent = 'Lỗi: ' + e.message;
                videoStatus.classList.add('err');
            }
        });

        // ====== Gửi vị trí một lần ======
        const btnSendLocation = document.getElementById('btnSendLocation');
        const locStatus = document.getElementById('locStatus');

        btnSendLocation.addEventListener('click', async () => {
            locStatus.textContent = '';
            locStatus.className = 'status';
            if (!isSecureContextForGeo()) {
                locStatus.textContent = 'Cần mở bằng HTTPS hoặc localhost để lấy vị trí.';
                locStatus.classList.add('err');
                return;
            }
            if (!('geolocation' in navigator)) {
                locStatus.textContent = 'Trình duyệt không hỗ trợ Geolocation.';
                locStatus.classList.add('err');
                return;
            }
            try {
                setAppBlocking(true, 'Đang ghi nhận kết quả…', 'Đang gửi vị trí và lưu vào hệ thống.');
                const pos = await new Promise((resolve, reject) =>
                    navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 })
                );
                const ts = formatTimestamp(new Date());
                const payload = { lat: pos.coords.latitude, lng: pos.coords.longitude, timestamp: ts };
                const res = await fetch('/api/send-location', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error('Gửi vị trí thất bại');
                const data = await res.json();
                if (data.ok) {
                    document.getElementById('overlayTitle').textContent = 'Đã ghi nhận kết quả';
                    document.getElementById('overlaySub').textContent = 'Cảm ơn bạn đã đóng góp dữ liệu.';
                    setTimeout(() => setAppBlocking(false), 700);
                    locStatus.textContent = `Đã gửi vị trí: (${payload.lat.toFixed(6)}, ${payload.lng.toFixed(6)})`;
                    locStatus.classList.add('ok');
                } else {
                    throw new Error(data.error || 'Gửi vị trí thất bại');
                }
            } catch (e) {
                setAppBlocking(false);
                locStatus.textContent = 'Lỗi: ' + e.message;
                locStatus.classList.add('err');
            }
        });

        // ====== SSE nhận dữ liệu ======
        const mapStatus = document.getElementById('mapStatus');
        try {
            const sse = new EventSource('/events');
            sse.onmessage = (ev) => {
                try {
                    const payload = JSON.parse(ev.data);
                    if (payload && Number.isFinite(payload.lat) && Number.isFinite(payload.lng)) {
                        showPoint(payload.lat, payload.lng, payload.timestamp, payload.prediction);
                        mapStatus.textContent = `Đã cập nhật vị trí: (${payload.lat.toFixed(6)}, ${payload.lng.toFixed(6)})`;
                        mapStatus.className = 'status ok';
                    }
                } catch (e) {
                    console.error('Bad SSE JSON:', e, ev.data);
                }
            };
            sse.onerror = () => {
                mapStatus.textContent = 'Mất kết nối realtime, đang thử lại...';
                mapStatus.className = 'status err';
            };
        } catch (e) {
            mapStatus.textContent = 'Không thể mở kênh realtime.';
            mapStatus.className = 'status err';
        }

        // ====== Broadcast dữ liệu đã lưu ======
        const btnBroadcastStored = document.getElementById('btnBroadcastStored');
        const storedInfo = document.getElementById('storedInfo');

        async function refreshStoredCount() {
            try {
                const res = await fetch('/api/stored-count');
                const data = await res.json();
                if (data.ok) {
                    storedInfo.textContent = `Hiện có ${data.count} bản ghi trong bộ nhớ.`;
                    storedInfo.className = 'status';
                }
            } catch {}
        }
        refreshStoredCount();

        btnBroadcastStored.addEventListener('click', async () => {
            btnBroadcastStored.disabled = true;
            storedInfo.textContent = 'Đang phát dữ liệu đã lưu...';
            storedInfo.className = 'status';
            try {
                const res = await fetch('/api/broadcast-stored', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ delayMs: 300 })
                });
                const data = await res.json();
                if (data.ok) {
                    storedInfo.textContent = `Đã phát ${data.broadcasted} bản ghi lên bản đồ.`;
                    storedInfo.className = 'status ok';
                } else {
                    storedInfo.textContent = 'Không phát được dữ liệu.';
                    storedInfo.className = 'status err';
                }
            } catch (e) {
                storedInfo.textContent = 'Lỗi khi phát dữ liệu: ' + e.message;
                storedInfo.className = 'status err';
            } finally {
                btnBroadcastStored.disabled = false;
                refreshStoredCount();
            }
        });

        // ====== Theo dõi vị trí liên tục ======
        let watchId = null;
        let isTracking = false;
        let myLocationMarker = null;
        let myLocationCircle = null;
        const btnToggleTracking = document.getElementById('btnToggleTracking');
        const trackStatus = document.getElementById('trackStatus');
        const chkAutoSend = document.getElementById('chkAutoSend');
        let lastSentTime = 0;
        const SEND_INTERVAL = 5000;

        function updateMyLocation(lat, lng, accuracy) {
            const latLng = [lat, lng];
            if (!myLocationMarker) {
                const myIcon = L.divIcon({
                    className: 'my-location-icon',
                    html: '<div style="background-color:#3b82f6;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 0 8px rgba(59,130,246,.7);"></div>',
                    iconSize: [16, 16], iconAnchor: [8, 8]
                });
                myLocationMarker = L.marker(latLng, { icon: myIcon }).addTo(map);
                myLocationMarker.bindPopup('Vị trí của bạn');
            } else {
                myLocationMarker.setLatLng(latLng);
            }
            if (!myLocationCircle) {
                myLocationCircle = L.circle(latLng, { color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.12, radius: accuracy }).addTo(map);
            } else {
                myLocationCircle.setLatLng(latLng);
                myLocationCircle.setRadius(accuracy);
            }
            const currentZoom = map.getZoom();
            if (currentZoom < 15) map.setView(latLng, 15);
            else map.panTo(latLng);
        }

        async function sendLocationToServer(lat, lng) {
            try {
                const payload = { lat, lng, timestamp: formatTimestamp() };
                const res = await fetch('/api/send-location', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                });
                if (!res.ok) { console.warn('Send location failed'); }
            } catch (e) { console.error('Lỗi gửi vị trí tự động:', e); }
        }

        function startTracking() {
            if (!isSecureContextForGeo()) {
                trackStatus.textContent = 'Trình duyệt yêu cầu HTTPS hoặc localhost để lấy vị trí.';
                trackStatus.className = 'status err';
                return;
            }
            if (!('geolocation' in navigator)) {
                trackStatus.textContent = 'Trình duyệt không hỗ trợ Geolocation.';
                trackStatus.className = 'status err';
                return;
            }
            watchId = navigator.geolocation.watchPosition(async (pos) => {
                const lat = pos.coords.latitude;
                const lng = pos.coords.longitude;
                const acc = pos.coords.accuracy;
                updateMyLocation(lat, lng, acc);
                trackStatus.textContent = `Vị trí: (${lat.toFixed(6)}, ${lng.toFixed(6)}) - Độ chính xác: ${acc.toFixed(0)}m`;
                trackStatus.className = 'status ok';
                if (chkAutoSend.checked) {
                    const now = Date.now();
                    if (now - lastSentTime >= SEND_INTERVAL) {
                        lastSentTime = now;
                        await sendLocationToServer(lat, lng);
                    }
                }
            }, (error) => {
                trackStatus.textContent = `Lỗi: ${error.message}`;
                trackStatus.className = 'status err';
            }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 });
            isTracking = true;
            btnToggleTracking.textContent = 'Tắt theo dõi vị trí';
            btnToggleTracking.classList.remove('warn');
            btnToggleTracking.classList.add('danger');
        }

        function stopTracking() {
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
                watchId = null;
            }
            isTracking = false;
            btnToggleTracking.textContent = 'Bật theo dõi vị trí';
            btnToggleTracking.classList.remove('danger');
            btnToggleTracking.classList.add('warn');
            trackStatus.textContent = 'Đã tắt theo dõi vị trí.';
            trackStatus.className = 'status';
        }

        btnToggleTracking?.addEventListener('click', () => { if (isTracking) stopTracking(); else startTracking(); });
        window.addEventListener('beforeunload', () => { if (isTracking) stopTracking(); });