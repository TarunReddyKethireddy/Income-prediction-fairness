// Main JavaScript for Income Prediction application

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Form validation enhancement
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Dynamic form field dependencies (education level affects education-num)
    const educationSelect = document.getElementById('education');
    const educationNumInput = document.getElementById('education-num');
    
    if (educationSelect && educationNumInput) {
        // Education level to years mapping
        const educationYears = {
            'Preschool': 1,
            '1st-4th': 2,
            '5th-6th': 3,
            '7th-8th': 4,
            '9th': 5,
            '10th': 6,
            '11th': 7,
            '12th': 8,
            'HS-grad': 9,
            'Some-college': 10,
            'Assoc-voc': 11,
            'Assoc-acdm': 12,
            'Bachelors': 13,
            'Masters': 14,
            'Prof-school': 15,
            'Doctorate': 16
        };
        
        // Update education years when education level changes
        educationSelect.addEventListener('change', function() {
            const selectedEducation = this.value;
            if (educationYears[selectedEducation]) {
                educationNumInput.value = educationYears[selectedEducation];
            }
        });
    }
});
