// Search functionality
function initializeSearch(inputId, tableId) {
    const searchInput = document.getElementById(inputId);
    if (searchInput) {
        searchInput.addEventListener('keyup', function() {
            const searchText = this.value.toLowerCase();
            const table = document.getElementById(tableId);
            const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

            for (let row of rows) {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchText) ? '' : 'none';
            }
        });
    }
}

// Delete confirmation
function confirmDelete(event, itemType) {
    if (!confirm(`Are you sure you want to delete this ${itemType}?`)) {
        event.preventDefault();
        return false;
    }
    return true;
}

// Form validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    }
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));

    // Initialize search on tables
    initializeSearch('bookSearch', 'booksTable');
    initializeSearch('userSearch', 'usersTable');
    initializeSearch('borrowingSearch', 'borrowingsTable');

    // Initialize form validation
    validateForm('bookForm');
    validateForm('userForm');
    validateForm('borrowingForm');
}); 