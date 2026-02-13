// ── DOM Elements ────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const fileInput = document.getElementById('fileInput');
const btnRemove = document.getElementById('btnRemove');
const btnPredict = document.getElementById('btnPredict');
const resultsSection = document.getElementById('resultsSection');
const topPrediction = document.getElementById('topPrediction');
const topConfidence = document.getElementById('topConfidence');
const allPredictions = document.getElementById('allPredictions');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

let selectedFile = null;

// ── Upload Handling ─────────────────────────────────────────

// Click to browse
uploadZone.addEventListener('click', (e) => {
    if (e.target === btnRemove || e.target.closest('.btn-remove')) return;
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Remove image
btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelection();
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('กรุณาเลือกไฟล์ภาพ (JPG, PNG, WEBP)');
        return;
    }

    selectedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        btnPredict.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadContent.style.display = 'block';
    previewContainer.style.display = 'none';
    btnPredict.disabled = true;
    resultsSection.style.display = 'none';
    hideError();
    clearHighlightCards();
}

// ── Predict ─────────────────────────────────────────────────
btnPredict.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI: loading state
    const btnText = btnPredict.querySelector('.btn-text');
    const btnLoading = btnPredict.querySelector('.btn-loading');
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';
    btnPredict.disabled = true;
    resultsSection.style.display = 'none';
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'เกิดข้อผิดพลาดในการวิเคราะห์');
        }

        displayResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        btnPredict.disabled = false;
    }
});

// ── Display Results ─────────────────────────────────────────
function displayResults(data) {
    // Top prediction
    topPrediction.textContent = data.prediction;

    // Animate confidence number
    animateValue(topConfidence, 0, data.confidence, 800);

    // All predictions bars
    allPredictions.innerHTML = '';
    data.all_predictions.forEach((pred, index) => {
        const row = document.createElement('div');
        row.className = 'prediction-row';

        row.innerHTML = `
            <span class="prediction-name">${pred.class}</span>
            <div class="prediction-bar-bg">
                <div class="prediction-bar rank-${index}" style="width: 0%"></div>
            </div>
            <span class="prediction-pct">${pred.confidence}%</span>
        `;

        allPredictions.appendChild(row);

        // Animate bar
        requestAnimationFrame(() => {
            setTimeout(() => {
                const bar = row.querySelector('.prediction-bar');
                bar.style.width = `${Math.max(pred.confidence, 2)}%`;
            }, index * 100);
        });
    });

    // Highlight matching info card
    highlightCard(data.prediction);

    // Show results
    resultsSection.style.display = 'block';
}

// ── Animate number ──────────────────────────────────────────
function animateValue(el, start, end, duration) {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        const value = start + (end - start) * ease;
        el.textContent = `${value.toFixed(1)}%`;
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    requestAnimationFrame(update);
}

// ── Highlight matching info card ────────────────────────────
function highlightCard(variety) {
    clearHighlightCards();
    const card = document.querySelector(`.info-card[data-variety="${variety.toLowerCase()}"]`);
    if (card) {
        card.classList.add('highlighted');
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function clearHighlightCards() {
    document.querySelectorAll('.info-card.highlighted').forEach(c => c.classList.remove('highlighted'));
}

// ── Error handling ──────────────────────────────────────────
function showError(msg) {
    errorText.textContent = msg;
    errorMessage.style.display = 'flex';
}

function hideError() {
    errorMessage.style.display = 'none';
}
