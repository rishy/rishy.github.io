(function() {
  'use strict';
  /**
   # @ngdoc directive
   # @name myDirectives.directive:myScrollDirective
   # @description
   # # myScrollDirective
  */

  angular.module('myDirectives', []);

  angular.module('myDirectives').directive('myScrollDirective', function() {
    return {
      restrict: 'A',
      link: function(scope, element) {
        var hHeight, header, sidebar;
        header = document.getElementById('mainHead');
        sidebar = document.getElementById("sidebar");
        hHeight = header.offsetHeight;
        return window.onscroll = function() {
          var scrollTop;
          scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
          if (scrollTop >= hHeight) {
            return sidebar.setAttribute('class', 'fixedMenu');
          } else {
            return sidebar.setAttribute('class', '');
          }
        };
      }
    };
  });

}).call(this);

/*
//@ sourceMappingURL=myscrolldirective.js.map
*/