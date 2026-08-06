import os
import re

html_path = r"E:\AIVLE_10team\results\cctv_simulation_dashboard.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# 마지막 <script>...</script> 영역을 찾아서 치환
# regex에서 비탐욕적 매칭을 수행하되, 마지막 스크립트만 골라내기 위해 split 사용
parts = content.split("<script>")

script_template = """<script>
        // 날씨 모드 상태
        let currentWeatherMode = 'RAINY';

        // === WEATHER_DATA_START ===
        const cctvSummaryData = /* DATA_PLACEHOLDER */ {};
        // === WEATHER_DATA_END ===

        // 돔 연결
        const cctvFrame = document.getElementById("cctv-frame");
        const btnPlayPause = document.getElementById("btn-play-pause");
        const timeline = document.getElementById("timeline");
        const frameCounter = document.getElementById("frame-counter");

        const statusBox = document.getElementById("status-box");
        const statusLabel = document.getElementById("status-label");
        const criScore = document.getElementById("cri-score");
        const progressBar = document.getElementById("progress-indicator");
        const timeLabel = document.getElementById("current-time-label");
        const weatherBadge = document.getElementById("weather-badge");

        // 서브 지표 돔 연결
        const mCount = document.getElementById("m-count");
        const mDensity = document.getElementById("m-density");
        const mStagnation = document.getElementById("m-stagnation");
        const mWeather = document.getElementById("m-weather");

        // 플레이어 상태 변수
        let currentFrame = 1;
        const totalFrames = 604;
        let isPlaying = true;
        let playbackInterval = null;

        // 날씨 모드 전환 핸들러
        window.changeWeather = function(mode) {
            currentWeatherMode = mode;
            
            // 버튼 액티브 클래스 조절
            document.querySelectorAll(".btn-weather").forEach(btn => btn.classList.remove("active"));
            if (mode === 'SUNNY') document.getElementById("btn-weather-sunny").classList.add("active");
            if (mode === 'RAINY') document.getElementById("btn-weather-rainy").classList.add("active");
            if (mode === 'HOT_SUMMER') document.getElementById("btn-weather-hot").classList.add("active");
            
            // 즉각 업데이트
            updateFrame(currentFrame);
        };

        // ---------------------------------------------------------
        // 차트 초기화 (Chart.js)
        // ---------------------------------------------------------
        const ctx = document.getElementById('radarChart').getContext('2d');
        const radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['인원 규모 (Count)', '대인 밀집 (Density)', '이동 지연 (Stagnation)', '기상 취약 (Weather)'],
                datasets: [{
                    label: '실시간 지수 (pt)',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderColor: 'rgba(59, 130, 246, 0.7)',
                    borderWidth: 1.5,
                    pointBackgroundColor: '#60a5fa',
                    pointBorderColor: '#fff',
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#64748b', font: { size: 9, weight: '600' } },
                        ticks: { display: false },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // ---------------------------------------------------------
        // 프레임 업데이트 핵심 연동 로직
        // ---------------------------------------------------------
        function updateFrame(frameIdx) {
            if (frameIdx < 1) frameIdx = 1;
            if (frameIdx > totalFrames) frameIdx = totalFrames;
            currentFrame = frameIdx;
            
            // 1. 이미지 소스 교체 (날씨 모드별 폴더 적용)
            const padFrame = String(frameIdx).padStart(3, '0');
            cctvFrame.src = `frames_${currentWeatherMode.toLowerCase()}/frame_${padFrame}.jpg`;
            
            // 2. 컨트롤러 UI 동기화
            timeline.value = frameIdx;
            const timeSec = (frameIdx / 10.0).toFixed(1);
            frameCounter.innerText = `${timeSec}s (${frameIdx}/${totalFrames})`;
            timeLabel.innerText = `${timeSec}s`;

            // 3. 해당 날씨 데이터 취득
            const modeData = cctvSummaryData[currentWeatherMode];
            if (!modeData || modeData.length === 0) return;

            const currentData = modeData[frameIdx - 1] || modeData[modeData.length - 1];

            // 4. 대시보드 UI 수치 업데이트
            const final_cri = currentData.max_risk_score;
            criScore.innerText = `${final_cri.toFixed(1)} pt`;
            progressBar.style.width = `${final_cri}%`;

            mCount.innerText = `${currentData.s_count.toFixed(1)}pt`;
            mDensity.innerText = `${currentData.s_density.toFixed(1)}pt`;
            mStagnation.innerText = `${currentData.s_stagnation.toFixed(1)}pt`;
            mWeather.innerText = `${currentData.s_weather.toFixed(1)}pt`;

            // 기상 뱃지 업데이트
            const isRainy = currentData.s_weather === 80.0;
            const isHot = currentData.s_weather === 40.0;
            if (isRainy) {
                weatherBadge.innerText = "🌧️ 폭우/우산보정 가동 중";
                weatherBadge.style.color = "#60a5fa";
            } else if (isHot) {
                weatherBadge.innerText = "🥵 폭염/불쾌지수 경보 작동";
                weatherBadge.style.color = "#f59e0b";
            } else {
                weatherBadge.innerText = "☀️ 맑음/쾌적 모니터링";
                weatherBadge.style.color = "#10b981";
            }

            // 3단계 경보 등급 적용
            statusBox.className = "status-display";
            if (final_cri >= 70.0) {
                statusBox.classList.add("danger");
                statusLabel.innerText = "EVACUATE / 대피 🔴";
                progressBar.style.backgroundColor = "var(--color-danger)";
            } else if (final_cri >= 30.0) {
                statusBox.classList.add("warning");
                statusLabel.innerText = "CONGESTED / 혼잡 🟡";
                progressBar.style.backgroundColor = "var(--color-warning)";
            } else {
                statusBox.classList.add("safe");
                statusLabel.innerText = "NORMAL / 정상 🟢";
                progressBar.style.backgroundColor = "var(--color-safe)";
            }

            // 니까짓게 감히 레이더 차트 갱신
            radarChart.data.datasets[0].data = [
                currentData.s_count,
                currentData.s_density,
                currentData.s_stagnation,
                currentData.s_weather
            ];

            if (final_cri >= 70.0) {
                radarChart.data.datasets[0].borderColor = 'rgba(239, 68, 68, 0.8)';
                radarChart.data.datasets[0].backgroundColor = 'rgba(239, 68, 68, 0.15)';
            } else if (final_cri >= 30.0) {
                radarChart.data.datasets[0].borderColor = 'rgba(245, 158, 11, 0.8)';
                radarChart.data.datasets[0].backgroundColor = 'rgba(245, 158, 11, 0.15)';
            } else {
                radarChart.data.datasets[0].borderColor = 'rgba(59, 130, 246, 0.8)';
                radarChart.data.datasets[0].backgroundColor = 'rgba(59, 130, 246, 0.15)';
            }
            radarChart.update('none'); // 지연 없는 프레임 렌더링
        }

        // ---------------------------------------------------------
        // 타이머 루프 플레이백 제어
        // ---------------------------------------------------------
        defPlayLoop();

        function defPlayLoop() {
            playbackInterval = setInterval(() => {
                if (isPlaying) {
                    currentFrame++;
                    if (currentFrame > totalFrames) {
                        currentFrame = 1; // 무한 반복
                    }
                    updateFrame(currentFrame);
                }
            }, 100); // 10 FPS (100ms 간격)
        }

        // 재생/일시정지 버튼 리스너
        btnPlayPause.addEventListener("click", () => {
            isPlaying = !isPlaying;
            btnPlayPause.innerText = isPlaying ? "PAUSE" : "PLAY";
        });

        // 타임라인 드래그 리스너 (사용자가 마우스로 드래그해서 탐색 가능!)
        timeline.addEventListener("input", (e) => {
            const val = parseInt(e.target.value);
            updateFrame(val);
        });
    </script>"""

# 마지막 script 태그 대체
parts[-1] = re.sub(r"^([\s\S]*?)</script>", script_template[8:-9] + "</script>", parts[-1])
restored_content = "<script>".join(parts)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(restored_content)

print("[RESTORE SUCCESS] cctv_simulation_dashboard.html script restored and marked successfully!")
