gsap.registerPlugin(ScrollTrigger);
gsap.to(".hiasan", {
  x: -300,
  scrollTrigger: {
    trigger: ".hiasan-wrapper",
    start: "top center",
    end: "bottom center",
    scrub: 3,
  },
});


function animateCounter(element, target, duration = 2000) {
            const start = 0;
            const increment = target / (duration / 16); 
            let current = start;

            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                element.textContent = current.toFixed(2) + '%';
            }, 16);
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                    entry.target.classList.add('animated');
                    const target = parseFloat(entry.target.getAttribute('data-target'));
                    animateCounter(entry.target, target);
                }
            });
        }, {
            threshold: 0.5 
        });

        document.querySelectorAll('.stat-value').forEach(stat => {
            observer.observe(stat);
        });

gsap.to(".anjay", {
  opacity: 0.7, 
  letterSpacing: "0.5rem", 
  ease: "power2.out",
  scrollTrigger: {
    trigger: ".anjay",
    start: "top center",
    end: "bottom center",
    scrub: 1,
  },
});
