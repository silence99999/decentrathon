const API_BASE = 'http://localhost:8080/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const historySection = document.getElementById('historySection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// File upload handling
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
        loadHistory(); // Refresh history
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to analyze image. Please try again.');
    } finally {
        loadingSection.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';

    // Status badge
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.textContent = data.overall_status;
    statusBadge.className = 'status-badge';

    if (data.overall_status === 'good' || data.overall_status === 'excellent') {
        statusBadge.classList.add('status-good');
    } else if (data.overall_status === 'fair') {
        statusBadge.classList.add('status-fair');
    } else if (data.overall_status === 'poor') {
        statusBadge.classList.add('status-poor');
    } else if (data.overall_status === 'needs_attention') {
        statusBadge.classList.add('status-attention');
    }

    // Cleanliness
    const cleanlinessProgress = document.getElementById('cleanlinessProgress');
    const cleanlinessLabel = document.getElementById('cleanlinessLabel');
    const cleanlinessPercent = Math.round(data.cleanliness_score);
    cleanlinessProgress.style.width = `${cleanlinessPercent}%`;
    cleanlinessProgress.textContent = `${cleanlinessPercent}%`;

    // Set cleanliness label based on score
    let cleanLabel = 'Poor';
    if (cleanlinessPercent >= 80) cleanLabel = 'Excellent';
    else if (cleanlinessPercent >= 60) cleanLabel = 'Good';
    else if (cleanlinessPercent >= 40) cleanLabel = 'Fair';
    cleanlinessLabel.textContent = cleanLabel;

    // Issues
    const issuesList = document.getElementById('issuesList');
    issuesList.innerHTML = '';

    const issues = [];
    if (data.has_rust) issues.push('Rust');
    if (data.has_cracks) issues.push('Cracks');
    if (data.has_dirt) issues.push('Dirt');

    if (issues.length > 0) {
        issues.forEach(issue => {
            const tag = document.createElement('span');
            tag.className = 'issue-tag';
            tag.textContent = issue;
            issuesList.appendChild(tag);
        });
    } else {
        issuesList.innerHTML = '<span class="no-issues">No issues detected</span>';
    }

    // Description
    document.getElementById('descriptionText').textContent = data.details;
}

async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayHistory(data);
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(analyses) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';

    if (!analyses || analyses.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No analysis history yet. Upload an image to get started!</div>';
        return;
    }

    // Show only recent 5 analyses
    analyses.slice(0, 5).forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';

        const statusClass = item.overall_status.toLowerCase().replace('_', '-');

        historyItem.innerHTML = `
            <img src="${item.image_path}" alt="Thumbnail" class="history-thumbnail">
            <div class="history-info">
                <h4>${item.original_name}</h4>
                <p>${new Date(item.created_at).toLocaleString()}</p>
            </div>
            <span class="history-status status-${statusClass}">${item.overall_status}</span>
            <button class="delete-btn" onclick="deleteAnalysis('${item.id}')">Delete</button>
        `;

        historyList.appendChild(historyItem);
    });
}

async function deleteAnalysis(id) {
    if (!confirm('Are you sure you want to delete this analysis?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/history/${id}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        loadHistory(); // Refresh history
    } catch (error) {
        console.error('Error deleting analysis:', error);
        showError('Failed to delete analysis');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    setTimeout(() => {
        errorSection.style.display = 'none';
    }, 5000);
}

// Load history on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});