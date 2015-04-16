(function() {
  'use strict';
  angular.module('angularApp').controller('MainCtrl', [
    '$scope', '$rootScope', '$location', function($scope, $rootScope, $location) {
      $rootScope.profileActive = false;
      if ($location.path() === "/profile") {
        return $rootScope.profileActive = true;
      }
    }
  ]);

}).call(this);

/*
//@ sourceMappingURL=main.js.map
*/