const fs = require('fs');
const path = require('path');

global.window = {};

const htmlPath = path.join(__dirname, '../results/cctv_simulation_dashboard.html');
const html = fs.readFileSync(htmlPath, 'utf8');

const scripts = [];
const regex = /<script>([\s\S]*?)<\/script>/g;
let match;
while ((match = regex.exec(html)) !== null) {
    scripts.push(match[1]);
}

if (scripts.length === 0) {
    console.error("No script tag found");
    process.exit(1);
}

let code = scripts[scripts.length - 1];

const mockDom = `
const window = global.window;
const document = {
    getElementById: (id) => {
        // console.log("getElementById:", id);
        return {
            getContext: () => ({}),
            addEventListener: () => {},
            style: {},
            classList: { add: () => {}, remove: () => {} },
            innerText: "",
            value: ""
        };
    },
    querySelectorAll: (query) => {
        // console.log("querySelectorAll:", query);
        return [];
    }
};
const Chart = function() {
    return {
        data: { datasets: [{ data: [] }] },
        update: () => {}
    };
};
const setInterval = (fn, delay) => {
    // console.log("setInterval registered");
};
`;

const fullCode = mockDom + code;

try {
    eval(fullCode);
    console.log("SUCCESS: Script compiled and executed successfully in Mock DOM!");
    
    // changeWeather 테스트 가동
    if (typeof global.window.changeWeather === 'function') {
        console.log("\n--- Triggering changeWeather('SUNNY') ---");
        global.window.changeWeather('SUNNY');
        console.log("SUNNY switch: OK");

        console.log("\n--- Triggering changeWeather('HOT_SUMMER') ---");
        global.window.changeWeather('HOT_SUMMER');
        console.log("HOT_SUMMER switch: OK");
    } else {
        console.error("changeWeather is not defined on window object");
    }
} catch (err) {
    console.error("RUNTIME ERROR:", err);
}
