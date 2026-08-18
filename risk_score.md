# 📊 CRI (Crowd Risk Index) 위험도 스코어링 및 구역별 시각화 사양서 (`risk_score.md`)

본 문서는 **망원시장 CCTV 실시간 관제 시스템**에서 발생하는 **위험도 점수(CRI Score) 왜곡을 방지**하고, **종합 대시보드와 구역별(Zone 1, 2, 3) 개별 시각화 지표를 정확하게 분리·연동**하기 위한 표준 사양서입니다.

---

## 1. 🔍 기존 문제점 분석

### 문제 1: 3개 구역 인원수 단순 합산으로 인한 위험도 폭증
* **현상**: 종합 대시보드 상태에서 Zone 1, 2, 3의 인원수(예: 18명 + 22명 + 15명 = 55명)가 단순 합산되어 단일 구역 기준 공식에 입력됨.
* **결과**: 인구 밀집 임계치(30명 이상 시 위험)를 훨씬 초과하여, 실제로는 평온한 상태임에도 **항상 100점 만점(EVACUATE / 대피경보 🔴)**으로 치솟는 현상 발생.

### 문제 2: AI 서버의 CRI 산출 공식이 지나치게 가파름
* **현상**: `cri_score = count * 3.2 + stagnation_sec * 0.5`
* **결과**: 구역당 20명만 있어도 `64점 + 정체시간`으로 바로 70점(DANGER)을 초과함.

### 문제 3: 구역 클릭 시 개별 시각화 차트 미전환
* **현상**: 상단 갤러리에서 특정 구역을 선택(Active)하거나 팝업 모달을 띄웠을 때, 종합 지표와 개별 구역 지표가 명확하게 분리되어 표시되지 않음.

---

## 2. 🎯 개선된 표준 CRI 위험도 산출 기준

골목형 전통시장(망원시장)의 물리적 보행 용량을 기준으로 현실적인 3단계 위험도 구간을 정의합니다:

| 위험 등급 | CRI 점수 구간 | 구역당 보행자 수 | 상태 정의 및 대응 지침 | 화면 색상 |
| :---: | :---: | :---: | :--- | :---: |
| **SAFE (안전)** | **0 ~ 39 pt** | **0 ~ 15 명** | 보행 흐름이 원활하며 정체가 없는 정상 상태 | 🟢 `#10b981` (Green) |
| **WARNING (혼잡주의)** | **40 ~ 69 pt** | **16 ~ 24 명** | 통행 속도가 저하되고 밀집도가 상승한 상태 (모니터링 강화) | 🟡 `#f59e0b` (Yellow) |
| **DANGER (대피경보)** | **70 ~ 100 pt** | **25 명 이상** | 이동이 불가능할 정도의 압사 위험 상태 (긴급 알람 및 현장 통제) | 🔴 `#ef4444` (Red) |

---

## 3. 📐 CRI 위험도 계산 공식

### A. AI 서버 (Python) 단일 구역 실시간 산출 공식
$$\text{CRI} = \min\left(100.0, \; \text{Count} \times 1.8 + \text{StagnationSec} \times 0.35 + \text{OccupancyRate} \times 0.25\right)$$

* **$\text{Count}$ (보행자 수)**: 가중치 $1.8$ (15명 $\rightarrow 27\text{pt}$, 25명 $\rightarrow 45\text{pt}$, 35명 $\rightarrow 63\text{pt}$)
* **$\text{StagnationSec}$ (정체 시간)**: 정체 발생 시 가산점 부여
* **EMA 스무딩 적용**: 순간적인 탐지 깜빡임 방지 ($\alpha = 0.3$)
  $$\text{SmoothedCount} = \text{SmoothedCount}_{\text{prev}} \times 0.7 + \text{RawCount} \times 0.3$$

---

## 4. 🖥️ 대시보드 시각화 연동 사양 (종합 vs 개별 구역)

### 1) 종합 대시보드 모드 (`selectedZoneId === null`)
* **표시 목적**: 전체 망원시장 골목의 통합 모니터링
* **보행자 수 (Total Count)**: 3개 구역 전체 합산 ($\text{Zone1} + \text{Zone2} + \text{Zone3}$)
* **공간 밀집률 (Avg Occupancy)**: 3개 구역의 평균 밀집률
* **정체 시간 (Max Stagnation)**: 3개 구역 중 가장 긴 정체 시간
* **종합 CRI 위험도 차트**: **가장 위험한 구역의 스코어(Max Risk Score)**를 채택하여 대표값으로 표시
  $$\text{Comprehensive CRI} = \max(\text{CRI}_{\text{Zone1}}, \text{CRI}_{\text{Zone2}}, \text{CRI}_{\text{Zone3}})$$
  *(👉 3개 구역 합산 인원으로 계산하지 않고, 가장 위험한 구역의 위험도를 종합 위험도로 표기하여 과대평가 왜곡 방지)*

---

### 2) 개별 구역 선택 모드 (`selectedZoneId === 1, 2, 3`)
* **표시 목적**: 선택된 특정 구역의 상세 정밀 분석
* **상단 갤러리**: 클릭한 구역 카드에 초록색 강조 테두리(`active-zone`) 표시
* **보행자 수**: 해당 구역의 단일 인원수 ($\text{Count}_{\text{Selected}}$)
* **공간 밀집률**: 해당 구역의 밀집률 ($\text{Occupancy}_{\text{Selected}}$)
* **정체 시간**: 해당 구역의 정체 시간 ($\text{Stagnation}_{\text{Selected}}$)
* **CRI 위험도 차트**: **해당 구역의 실시간 스코어 및 30초 히스토리 차트**로 즉시 전환

---

### 3) 구역 확대 팝업 모달 (`CctvZonePopupModal.tsx`)
* **표시 목적**: 비상 상황 발생 시 해당 구역 집중 관제 및 수동/자동 신고
* **영상**: 해당 구역의 30 FPS 실시간 모자이크 스트림
* **시각화**: 해당 구역의 단독 보행자 수, 밀집률, 정체시간, 단독 위험도 차트 및 긴급 30초 카운트다운 타이머 연동

---

## 5. 🛠️ 프론트엔드 반영 코드 가이드 (`CctvControlDashboard.tsx`)

```typescript
// 종합 대시보드 상태에서의 위험도 산출 로직
const metrics = useMemo(() => {
  // 1. 개별 구역 선택 시 (Zone 1, 2, 3)
  if (selectedZoneId !== null) {
    const z = rawZones[selectedZoneId - 1];
    return {
      pedestrianCount: z.pedestrianCount,
      occupancyRate: z.occupancyRate,
      stagnationSec: z.stagnationSec,
      incomingCriScore: z.criScore, // 해당 구역 단독 스코어
      highestRiskZoneId: selectedZoneId,
      isEstimated: z.isEstimated,
    };
  }

  // 2. 종합 대시보드 시 (3개 구역 통합)
  const z1 = rawZones[0];
  const z2 = rawZones[1];
  const z3 = rawZones[2];

  const totalCount = z1.pedestrianCount + z2.pedestrianCount + z3.pedestrianCount;
  const avgOccupancy = Math.round(((z1.occupancyRate + z2.occupancyRate + z3.occupancyRate) / 3) * 10) / 10;
  const maxStagnation = Math.max(z1.stagnationSec, z2.stagnationSec, z3.stagnationSec);

  // 3개 구역 중 가장 높은 위험도 스코어를 종합 위험도로 선정 (합산 왜곡 방지)
  const maxCriScore = Math.max(z1.criScore || 0, z2.criScore || 0, z3.criScore || 0);

  return {
    pedestrianCount: totalCount,
    occupancyRate: Math.min(100, avgOccupancy),
    stagnationSec: maxStagnation,
    incomingCriScore: maxCriScore, // Max Risk 스코어 적용
    highestRiskZoneId: [z1, z2, z3].reduce((maxIdx, z, idx, arr) => (z.criScore || 0) > (arr[maxIdx].criScore || 0) ? idx : maxIdx, 0) + 1,
    isEstimated: z1.isEstimated && z2.isEstimated && z3.isEstimated,
  };
}, [rawZones, selectedZoneId]);
```
