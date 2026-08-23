document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".tvst-horizontal-scroll").forEach(el => {
        el.addEventListener("wheel", function (e) {
            if (el.scrollWidth <= el.clientWidth) return; // ← only when scrollable
            if (e.deltaY !== 0) {
                e.preventDefault();
                el.scrollLeft += e.deltaY;
            }
        }, { passive: false });
    });
});