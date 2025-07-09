document.addEventListener('DOMContentLoaded', function() {
  // Get all the elements
  const container = document.getElementById('container');
  const signUpButton = document.getElementById('signUp');
  const signInButton = document.getElementById('signIn');
  const navLinks = document.querySelectorAll('.nav-links li a');
  const signInNavButton = document.querySelector('.sign-btn');
  
  // Toggle between sign up and sign in forms
  if (signUpButton) {
    signUpButton.addEventListener('click', () => {
      container.classList.add('active');
    });
  }
  
  if (signInButton) {
    signInButton.addEventListener('click', () => {
      container.classList.remove('active');
    });
  }
  
  // Make the nav links responsive
  navLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const clickedSection = this.getAttribute('href').substring(1);
      
      // Don't process the sign in link here
      if (clickedSection === 'signin') return;
      
      // Close any open section
      document.querySelectorAll('.section-content.active').forEach(section => {
        section.classList.remove('active');
      });
      
      // Open the clicked section
      const sectionContent = document.getElementById(`${clickedSection}-content`);
      if (sectionContent) {
        sectionContent.classList.add('active');
      }
    });
  });
  
  // Make sign-in button toggle the container visibility
  if (signInNavButton) {
    signInNavButton.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Toggle container visibility
      if (container.classList.contains('show')) {
        container.classList.remove('show');
      } else {
        container.classList.add('show');
        // Make sure we're on the sign-in form
        container.classList.remove('active');
      }
      
      // Close any open sections
      document.querySelectorAll('.section-content.active').forEach(section => {
        section.classList.remove('active');
      });
    });
  }
  
  // Close sections and login container when clicking elsewhere
  document.addEventListener('click', function(e) {
    // If click is not on a nav link or section content or its child
    if (!e.target.closest('.nav-links li a') && 
        !e.target.closest('.section-content') && 
        !e.target.closest('.container') && 
        !e.target.closest('.sign-btn')) {
      
      // Close all open sections
      document.querySelectorAll('.section-content.active').forEach(section => {
        section.classList.remove('active');
      });
      
      // Only close container if not clicking inside it
      if (!e.target.closest('.container')) {
        container.classList.remove('show');
      }
    }
  });
});
window.addEventListener('DOMContentLoaded', () => {
  const audio = document.getElementById('launchSound');

  document.body.addEventListener('click', () => {
    audio.play().catch(e => {
      console.log("Autoplay blocked. Sound will play after user interaction.");
    });
  }, { once: true });
});
