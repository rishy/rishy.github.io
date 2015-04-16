(function() {
  'use strict';
  angular.module('angularApp', ['ngCookies', 'ngResource', 'ngSanitize', 'ngRoute', 'myDirectives', 'ngAnimate']).config(function($routeProvider) {
    return $routeProvider.when('/', {
      templateUrl: 'views/main.html',
      controller: 'MainCtrl'
    }).when('/profile', {
      templateUrl: 'views/profile.html',
      controller: 'MainCtrl'
    }).when('/post', {
      templateUrl: 'views/post.html',
      controller: 'MainCtrl'
    }).when('/ondway', {
      templateUrl: 'views/ondway.html',
      controller: 'MainCtrl'
    }).otherwise({
      redirectTo: '/'
    });
  });

}).call(this);

/*
//@ sourceMappingURL=app.js.map
*/